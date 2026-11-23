#pragma once

#include <fhelium/rns.h>

// Backend entry points behind the public functions of the same name.
// The cpu namespace is compiled from cpu/ sources; the gpu namespace is
// compiled from either cuda/ or hip/ sources.

namespace fhelium::rns::cpu {
decltype(::fhelium::rns::montgomery_mul) montgomery_mul;
decltype(::fhelium::rns::montgomery_weighted_sum) montgomery_weighted_sum;
decltype(::fhelium::rns::montgomery_weighted_sums) montgomery_weighted_sums;
decltype(::fhelium::rns::montgomery_mul_cyclic_compressed)
    montgomery_mul_cyclic_compressed;
decltype(::fhelium::rns::montgomery_mul_contiguous_compressed)
    montgomery_mul_contiguous_compressed;
decltype(::fhelium::rns::montgomery_mul_row_scalars_)
    montgomery_mul_row_scalars_;
decltype(::fhelium::rns::to_montgomery_) to_montgomery_;
decltype(::fhelium::rns::from_montgomery_) from_montgomery_;
decltype(::fhelium::rns::add_lazy) add_lazy;
decltype(::fhelium::rns::add_lazy_with_twice_modulus)
    add_lazy_with_twice_modulus;
decltype(::fhelium::rns::sub_lazy) sub_lazy;
decltype(::fhelium::rns::add_standard) add_standard;
decltype(::fhelium::rns::add_standard_) add_standard_;
decltype(::fhelium::rns::sum_standard_batch) sum_standard_batch;
decltype(::fhelium::rns::sub_standard) sub_standard;
decltype(::fhelium::rns::sub_standard_) sub_standard_;
decltype(::fhelium::rns::montgomery_mul_row_scalars_standard)
    montgomery_mul_row_scalars_standard;
decltype(::fhelium::rns::reduce_to_standard_) reduce_to_standard_;
decltype(::fhelium::rns::center_residues_) center_residues_;
decltype(::fhelium::rns::shift_residues_positive_) shift_residues_positive_;
decltype(::fhelium::rns::lift_integer_coefficients) lift_integer_coefficients;
decltype(::fhelium::rns::mixed_radix_decompose) mixed_radix_decompose;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery)
    mixed_radix_basis_extend_to_montgomery;
decltype(::fhelium::rns::mixed_radix_decompose_out_) mixed_radix_decompose_out_;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery_out_)
    mixed_radix_basis_extend_to_montgomery_out_;
}  // namespace fhelium::rns::cpu

namespace fhelium::rns::gpu {
decltype(::fhelium::rns::montgomery_mul) montgomery_mul;
decltype(::fhelium::rns::montgomery_weighted_sum) montgomery_weighted_sum;
decltype(::fhelium::rns::montgomery_weighted_sums) montgomery_weighted_sums;
decltype(::fhelium::rns::montgomery_mul_cyclic_compressed)
    montgomery_mul_cyclic_compressed;
decltype(::fhelium::rns::montgomery_mul_contiguous_compressed)
    montgomery_mul_contiguous_compressed;
decltype(::fhelium::rns::montgomery_mul_row_scalars_)
    montgomery_mul_row_scalars_;
decltype(::fhelium::rns::to_montgomery_) to_montgomery_;
decltype(::fhelium::rns::from_montgomery_) from_montgomery_;
decltype(::fhelium::rns::add_lazy) add_lazy;
decltype(::fhelium::rns::add_lazy_with_twice_modulus)
    add_lazy_with_twice_modulus;
decltype(::fhelium::rns::sub_lazy) sub_lazy;
decltype(::fhelium::rns::add_standard) add_standard;
decltype(::fhelium::rns::add_standard_) add_standard_;
decltype(::fhelium::rns::sum_standard_batch) sum_standard_batch;
decltype(::fhelium::rns::sub_standard) sub_standard;
decltype(::fhelium::rns::sub_standard_) sub_standard_;
decltype(::fhelium::rns::montgomery_mul_row_scalars_standard)
    montgomery_mul_row_scalars_standard;
decltype(::fhelium::rns::reduce_to_standard_) reduce_to_standard_;
decltype(::fhelium::rns::center_residues_) center_residues_;
decltype(::fhelium::rns::shift_residues_positive_) shift_residues_positive_;
decltype(::fhelium::rns::lift_integer_coefficients) lift_integer_coefficients;
decltype(::fhelium::rns::mixed_radix_decompose) mixed_radix_decompose;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery)
    mixed_radix_basis_extend_to_montgomery;
decltype(::fhelium::rns::mixed_radix_decompose_out_) mixed_radix_decompose_out_;
decltype(::fhelium::rns::mixed_radix_basis_extend_to_montgomery_out_)
    mixed_radix_basis_extend_to_montgomery_out_;
// CUDA builds only.
decltype(::fhelium::rns::crt_basis_extend_out_) crt_basis_extend_out_;
decltype(::fhelium::rns::crt_basis_extend) crt_basis_extend;
decltype(::fhelium::rns::mixed_radix_modup_tiled) mixed_radix_modup_tiled;
decltype(::fhelium::rns::mixed_radix_modup_tiled_out_)
    mixed_radix_modup_tiled_out_;
}  // namespace fhelium::rns::gpu
