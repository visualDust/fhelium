"""Construct and execute CKKS operations with runtime-owned resources.

`Engine` accepts a CKKS configuration and creates device-local arithmetic and
random resources when a call first selects each device. Source factories use
PyTorch's default device unless the caller supplies `device`. The Engine owns
key lifecycles and does not copy key material across devices unless automatic
replication is enabled. Evaluator calls dispatch from operand placement, update
public metadata, and invoke registered Tensor operations. Program compilation
and execution belong to Backend and its Compile/JIT callers.
"""

from ._engine import Engine

__all__ = ["Engine"]
