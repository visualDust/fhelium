#pragma once

#include <fhelium/ckks.h>

// Backend entry points behind the public functions of the same name.
// The cpu namespace is compiled from cpu/ sources; the gpu namespace is
// compiled from either cuda/ or hip/ sources.

namespace fhelium::ckks::cpu {
decltype(::fhelium::ckks::multiply_two_component_ntt_montgomery_out)
    multiply_two_component_ntt_montgomery_out;
decltype(::fhelium::ckks::add_prepared_plaintext_component_out)
    add_prepared_plaintext_component_out;
decltype(::fhelium::ckks::add_cyclic_compressed_plaintext_component_out)
    add_cyclic_compressed_plaintext_component_out;
decltype(::fhelium::ckks::add_contiguous_compressed_plaintext_component_out)
    add_contiguous_compressed_plaintext_component_out;
decltype(::fhelium::ckks::add_strided_plaintext_component_out)
    add_strided_plaintext_component_out;
decltype(::fhelium::ckks::apply_coefficient_galois_automorphism_out)
    apply_coefficient_galois_automorphism_out;
decltype(::fhelium::ckks::apply_ntt_galois_automorphism_out)
    apply_ntt_galois_automorphism_out;
decltype(::fhelium::ckks::add_ntt_galois_automorphism_)
    add_ntt_galois_automorphism_;
decltype(::fhelium::ckks::keyswitch_moddown_qp_to_q_out)
    keyswitch_moddown_qp_to_q_out;
decltype(::fhelium::ckks::keyswitch_accumulate_digit_products_)
    keyswitch_accumulate_digit_products_;
}  // namespace fhelium::ckks::cpu

namespace fhelium::ckks::gpu {
decltype(::fhelium::ckks::multiply_two_component_ntt_montgomery_out)
    multiply_two_component_ntt_montgomery_out;
decltype(::fhelium::ckks::add_prepared_plaintext_component_out)
    add_prepared_plaintext_component_out;
decltype(::fhelium::ckks::add_cyclic_compressed_plaintext_component_out)
    add_cyclic_compressed_plaintext_component_out;
decltype(::fhelium::ckks::add_contiguous_compressed_plaintext_component_out)
    add_contiguous_compressed_plaintext_component_out;
decltype(::fhelium::ckks::add_strided_plaintext_component_out)
    add_strided_plaintext_component_out;
decltype(::fhelium::ckks::apply_coefficient_galois_automorphism_out)
    apply_coefficient_galois_automorphism_out;
decltype(::fhelium::ckks::apply_ntt_galois_automorphism_out)
    apply_ntt_galois_automorphism_out;
decltype(::fhelium::ckks::add_ntt_galois_automorphism_)
    add_ntt_galois_automorphism_;
decltype(::fhelium::ckks::keyswitch_moddown_qp_to_q_out)
    keyswitch_moddown_qp_to_q_out;
decltype(::fhelium::ckks::keyswitch_accumulate_digit_products_)
    keyswitch_accumulate_digit_products_;
decltype(::fhelium::ckks::keyswitch_accumulate_products_)
    keyswitch_accumulate_products_;
}  // namespace fhelium::ckks::gpu
