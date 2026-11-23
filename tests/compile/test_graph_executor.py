"""Automatic graphs preserve prepared fallback and callable output structure."""

from types import SimpleNamespace

import pytest
import torch

from fhelium.compile import _graph
from fhelium.compile._graph import _GraphExecutor


@pytest.mark.parametrize("graph", ["auto", True, False])
def test_cpu_fallback_and_disabled_execution(graph):
    specialization = SimpleNamespace(
        signature="cpu", _execute=lambda values: [values[0] + values[1], None]
    )
    executor = _GraphExecutor(graph=graph)
    for value in (1.0, 4.0):
        result = executor.execute(specialization, (torch.tensor(value), 2))
        torch.testing.assert_close(result[0], torch.tensor(value + 2))
        assert result[1] is None
    (stats,) = executor.stats()
    assert stats["state"] == ("disabled" if graph is False else "fallback")
    assert stats["prepared_calls"] == 2
    assert stats["replays"] == 0
    assert (
        stats["fallback_reason"] is None
        if graph is False
        else ("non-CUDA" in stats["fallback_reason"])
    )
    executor.clear()
    assert executor.stats() == ()


def test_capture_failure_is_recorded_once_and_preserves_caller_context(
    monkeypatch,
):
    calls = []
    attempts = []
    closed = []

    def execute(values):
        calls.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        return {"value": values[0] * values[1], "literal": (None, [3, "ok"])}

    class Pool:
        def __init__(self, device):
            self.device = device

        def close(self):
            closed.append(self.device)

    def capture(function, *, example_inputs, warmup, pool):
        attempts.append((warmup, pool))
        # The prepared adapter receives the unchanged scalar, not a buffer leaf.
        (result,) = function(*example_inputs)
        torch.testing.assert_close(result, example_inputs[0] * 2)
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr(
        _graph, "_input_device", lambda values: torch.device("cuda:0")
    )
    monkeypatch.setattr(_graph, "CudaGraphPool", Pool)
    monkeypatch.setattr(_graph.CudaGraphProgram, "capture", capture)
    specialization = SimpleNamespace(signature="capture", _execute=execute)
    executor = _GraphExecutor()
    with torch.no_grad():
        for value in (1.0, 2.0, 4.0):
            result = executor.execute(specialization, (torch.tensor(value), 2))
            assert result["literal"] == (None, [3, "ok"])
            torch.testing.assert_close(result["value"], torch.tensor(value * 2))
    assert len(attempts) == 1
    assert attempts[0][0] == 1
    assert calls == [(False, False)] * 4
    (stats,) = executor.stats()
    assert stats["state"] == "fallback"
    assert stats["fallback_reason"] == "RuntimeError: capture unavailable"
    assert stats["prepared_calls"] == 3
    with torch.inference_mode():
        executor.execute(specialization, (torch.tensor(1.0), 2))
    assert len(executor.stats()) == 2
    assert calls[-1] == (False, True)
    executor.clear()
    assert closed == [torch.device("cuda:0")]


def test_prepared_errors_propagate_and_autograd_stays_prepared(monkeypatch):
    def fail(values):
        raise ValueError("prepared error")

    executor = _GraphExecutor()
    with pytest.raises(ValueError, match="prepared error"):
        executor.execute(SimpleNamespace(signature="error", _execute=fail), ())
    monkeypatch.setattr(
        _graph, "_input_device", lambda values: torch.device("cuda:0")
    )
    specialization = SimpleNamespace(
        signature="grad", _execute=lambda values: values[0] ** 2
    )
    source = torch.tensor(3.0, requires_grad=True)
    for _ in range(2):
        executor.execute(specialization, (source,)).backward()
    torch.testing.assert_close(source.grad, torch.tensor(12.0))
    assert (
        executor.stats()[-1]["fallback_reason"]
        == "Autograd outputs require prepared execution"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("copy_output", [True, False])
def test_cuda_graph_replays_nested_outputs_and_shares_pool(copy_output):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    executor = _GraphExecutor(copy_output=copy_output)

    def execute(values):
        result = values[0] + values[1]
        return {"value": result, "nested": ([result], None, "static")}

    first = SimpleNamespace(signature="first", _execute=execute)
    second = SimpleNamespace(signature="second", _execute=execute)
    x = torch.tensor([1.0, 2.0], device="cuda:0")
    try:
        warm = executor.execute(first, (x, 2))
        assert executor.stats()[0]["state"] == "warmed"
        saved = executor.execute(first, (x, 2))
        original = saved["value"].clone()
        assert saved["nested"][1:] == (None, "static")
        executor.execute(second, (x, 3))
        executor.execute(second, (x, 3))
        torch.testing.assert_close(saved["value"], original)
        assert len(executor._pools) == 1
        updated = executor.execute(first, (x + 5, 2))
        torch.testing.assert_close(updated["value"], x + 7)
        torch.testing.assert_close(warm["value"], x + 2)
        torch.testing.assert_close(
            saved["value"], original if copy_output else x + 7
        )
        assert [stats["state"] for stats in executor.stats()] == [
            "captured",
            "captured",
        ]
    finally:
        executor.clear()
    assert executor.stats() == ()
