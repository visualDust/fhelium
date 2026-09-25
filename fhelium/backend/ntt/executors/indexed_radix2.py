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


def forward_small_to_montgomery(
    operand, even_indices, odd_indices, twiddles, params
):
    """Evaluate a short indexed NTT with the existing modular primitives.

    Native NTT entry points require at least 256 coefficients. Short periodic
    plaintexts use the same indexed butterflies without padding the polynomial
    or changing its roots. Each stage computes (a+b*w, a-b*w) modulo each row.
    """
    from fhelium.native.wrapper import rns_ops

    result = operand.clone()
    rns_ops.to_montgomery_(result, params)
    for stage in range(even_indices.size(0)):
        even = even_indices[stage].long()
        odd = odd_indices[stage].long()
        a = result.index_select(-1, even)
        b = rns_ops.montgomery_mul(
            result.index_select(-1, odd),
            twiddles[:, stage, :].contiguous(),
            params,
        )
        result.index_copy_(-1, even, rns_ops.add_standard(a, b, params))
        result.index_copy_(-1, odd, rns_ops.sub_standard(a, b, params))
    return result
