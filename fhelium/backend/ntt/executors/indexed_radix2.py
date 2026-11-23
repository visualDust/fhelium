"""Prepare native indexed radix2 calls over Tensor operands."""

from fhelium.native.wrapper import ntt_ops

_OPERATIONS = {
    'forward_montgomery_': ntt_ops.forward_ntt_montgomery_indexed_,
    'forward_to_montgomery_': ntt_ops.forward_ntt_to_montgomery_indexed_,
    'forward_to_montgomery': ntt_ops.forward_ntt_to_montgomery_indexed,
    'inverse_montgomery_': ntt_ops.inverse_ntt_montgomery_indexed_,
    'inverse_to_standard_lazy_': ntt_ops.inverse_ntt_to_standard_lazy_indexed_,
    'inverse_to_standard_': ntt_ops.inverse_ntt_to_standard_indexed_,
    'inverse_to_centered_': ntt_ops.inverse_ntt_to_centered_indexed_,
}


def prepare_transition(transition: str):
    """Bind the native operation without retaining its Tensor inputs."""
    operation = _OPERATIONS[transition]
    if transition == "forward_to_montgomery":
        return operation

    def execute(operand, even_indices, odd_indices, twiddles, params):
        operation(operand, even_indices, odd_indices, twiddles, params)
        return operand

    return execute
