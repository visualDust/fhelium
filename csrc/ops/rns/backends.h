#pragma once

#include <fhelium/rns.h>

// Backend entry points behind the public functions of the same name.
// The cpu namespace is compiled from cpu/ sources; the gpu namespace is
// compiled from either cuda/ or hip/ sources.

namespace fhelium::rns::cpu {
decltype(::fhelium::rns::montgomery_mul) montgomery_mul;
decltype(::fhelium::rns::montgomery_weighted_sums_out)
    montgomery_weighted_sums_out;
decltype(::fhelium::rns::montgomery_mul_cyclic_compressed_out)
    montgomery_mul_cyclic_compressed_out;
decltype(::fhelium::rns::montgomery_mul_contiguous_compressed_out)
    montgomery_mul_contiguous_compressed_out;
decltype(::fhelium::rns::montgomery_mul_row_scalars_)
    montgomery_mul_row_scalars_;
decltype(::fhelium::rns::to_montgomery_) to_montgomery_;
decltype(::fhelium::rns::from_montgomery_) from_montgomery_;
decltype(::fhelium::rns::add_lazy_out) add_lazy_out;
decltype(::fhelium::rns::montgomery_mul_out) montgomery_mul_out;
decltype(::fhelium::rns::montgomery_weighted_sum_out)
    montgomery_weighted_sum_out;
decltype(::fhelium::rns::add_lazy_with_twice_modulus_out)
    add_lazy_with_twice_modulus_out;
decltype(::fhelium::rns::sub_lazy_out) sub_lazy_out;
decltype(::fhelium::rns::add_standard_out) add_standard_out;
decltype(::fhelium::rns::sub_standard_out) sub_standard_out;
decltype(::fhelium::rns::negate_standard_out) negate_standard_out;
decltype(::fhelium::rns::sum_standard_batch_out) sum_standard_batch_out;
decltype(::fhelium::rns::montgomery_mul_row_scalars_standard_out)
    montgomery_mul_row_scalars_standard_out;
decltype(::fhelium::rns::reduce_to_standard_) reduce_to_standard_;
decltype(::fhelium::rns::center_residues_) center_residues_;
decltype(::fhelium::rns::shift_residues_positive_) shift_residues_positive_;
decltype(::fhelium::rns::lift_integer_coefficients) lift_integer_coefficients;
decltype(::fhelium::rns::mixed_radix_decompose) mixed_radix_decompose;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery)
    mixed_radix_basis_extend_to_montgomery;
decltype(::fhelium::rns::mixed_radix_decompose_out) mixed_radix_decompose_out;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery_out)
    mixed_radix_basis_extend_to_montgomery_out;
}  // namespace fhelium::rns::cpu

namespace fhelium::rns::gpu {
decltype(::fhelium::rns::montgomery_mul) montgomery_mul;
decltype(::fhelium::rns::montgomery_weighted_sums_out)
    montgomery_weighted_sums_out;
decltype(::fhelium::rns::montgomery_mul_cyclic_compressed_out)
    montgomery_mul_cyclic_compressed_out;
decltype(::fhelium::rns::montgomery_mul_contiguous_compressed_out)
    montgomery_mul_contiguous_compressed_out;
decltype(::fhelium::rns::montgomery_mul_row_scalars_)
    montgomery_mul_row_scalars_;
decltype(::fhelium::rns::to_montgomery_) to_montgomery_;
decltype(::fhelium::rns::from_montgomery_) from_montgomery_;
decltype(::fhelium::rns::add_lazy_out) add_lazy_out;
decltype(::fhelium::rns::montgomery_mul_out) montgomery_mul_out;
decltype(::fhelium::rns::montgomery_weighted_sum_out)
    montgomery_weighted_sum_out;
decltype(::fhelium::rns::add_lazy_with_twice_modulus_out)
    add_lazy_with_twice_modulus_out;
decltype(::fhelium::rns::sub_lazy_out) sub_lazy_out;
decltype(::fhelium::rns::add_standard_out) add_standard_out;
decltype(::fhelium::rns::sub_standard_out) sub_standard_out;
decltype(::fhelium::rns::negate_standard_out) negate_standard_out;
decltype(::fhelium::rns::sum_standard_batch_out) sum_standard_batch_out;
decltype(::fhelium::rns::montgomery_mul_row_scalars_standard_out)
    montgomery_mul_row_scalars_standard_out;
decltype(::fhelium::rns::reduce_to_standard_) reduce_to_standard_;
decltype(::fhelium::rns::center_residues_) center_residues_;
decltype(::fhelium::rns::shift_residues_positive_) shift_residues_positive_;
decltype(::fhelium::rns::lift_integer_coefficients) lift_integer_coefficients;
decltype(::fhelium::rns::mixed_radix_decompose) mixed_radix_decompose;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery)
    mixed_radix_basis_extend_to_montgomery;
decltype(::fhelium::rns::mixed_radix_decompose_out) mixed_radix_decompose_out;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery_out)
    mixed_radix_basis_extend_to_montgomery_out;
// CUDA builds only.
decltype(::fhelium::rns::crt_basis_extend_out) crt_basis_extend_out;
decltype(::fhelium::rns::crt_basis_extend) crt_basis_extend;
decltype(::fhelium::rns::mixed_radix_modup_tiled) mixed_radix_modup_tiled;
decltype(::fhelium::rns::mixed_radix_modup_tiled_out)
    mixed_radix_modup_tiled_out;
}  // namespace fhelium::rns::gpu
