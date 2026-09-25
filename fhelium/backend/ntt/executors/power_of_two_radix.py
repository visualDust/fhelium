"""Prepare native power of two radix calls over Tensor operands."""

from fhelium.native.wrapper import ntt_ops

_OPERATIONS = {
    'forward_montgomery_': ntt_ops.forward_ntt_montgomery_power_of_two_radix_compact_,
    'forward_to_montgomery_': ntt_ops.forward_ntt_to_montgomery_power_of_two_radix_compact_,
    'forward_to_montgomery': ntt_ops.forward_ntt_to_montgomery_power_of_two_radix_compact,
    'inverse_montgomery_': ntt_ops.inverse_ntt_montgomery_power_of_two_radix_compact_,
    'inverse_to_standard_lazy_': ntt_ops.inverse_ntt_to_standard_lazy_power_of_two_radix_compact_,
    'inverse_to_standard_': ntt_ops.inverse_ntt_to_standard_power_of_two_radix_compact_,
    'inverse_to_centered_': ntt_ops.inverse_ntt_to_centered_power_of_two_radix_compact_,
}


def prepare_transition(transition: str):
    """Bind the native operation without retaining its Tensor inputs."""
    operation = _OPERATIONS[transition]
    if transition == "forward_to_montgomery":
        return operation

    def execute(operand, outer, roots, params):
        operation(operand, outer, roots, params)
        return operand

    return execute
