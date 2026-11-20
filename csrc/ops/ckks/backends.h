#pragma once

#include <fhelium/ckks.h>

// Backend entry points behind the public functions of the same name.
// The cpu namespace is compiled from cpu/ sources; the gpu namespace is
// compiled from either cuda/ or hip/ sources.

namespace fhelium::ckks::cpu {
decltype(::fhelium::ckks::multiply_two_component_ntt_montgomery)
    multiply_two_component_ntt_montgomery;
decltype(::fhelium::ckks::add_prepared_plaintext_component)
    add_prepared_plaintext_component;
decltype(::fhelium::ckks::add_prepared_plaintext_component_)
    add_prepared_plaintext_component_;
decltype(::fhelium::ckks::add_cyclic_compressed_plaintext_component)
    add_cyclic_compressed_plaintext_component;
decltype(::fhelium::ckks::add_cyclic_compressed_plaintext_component_)
    add_cyclic_compressed_plaintext_component_;
decltype(::fhelium::ckks::add_contiguous_compressed_plaintext_component)
    add_contiguous_compressed_plaintext_component;
decltype(::fhelium::ckks::add_contiguous_compressed_plaintext_component_)
    add_contiguous_compressed_plaintext_component_;
decltype(::fhelium::ckks::add_strided_plaintext_component)
    add_strided_plaintext_component;
decltype(::fhelium::ckks::add_strided_plaintext_component_)
    add_strided_plaintext_component_;
decltype(::fhelium::ckks::rescale_drop_leading_prime_nearest)
    rescale_drop_leading_prime_nearest;
decltype(::fhelium::ckks::rescale_drop_leading_prime_nearest_)
    rescale_drop_leading_prime_nearest_;
decltype(::fhelium::ckks::rescale_drop_leading_prime_truncate)
    rescale_drop_leading_prime_truncate;
decltype(::fhelium::ckks::rescale_drop_leading_prime_truncate_)
    rescale_drop_leading_prime_truncate_;
decltype(::fhelium::ckks::apply_coefficient_galois_automorphism)
    apply_coefficient_galois_automorphism;
decltype(::fhelium::ckks::apply_ntt_galois_automorphism)
    apply_ntt_galois_automorphism;
decltype(::fhelium::ckks::keyswitch_moddown_qp_to_q) keyswitch_moddown_qp_to_q;
decltype(::fhelium::ckks::keyswitch_accumulate_digit_products_)
    keyswitch_accumulate_digit_products_;
}  // namespace fhelium::ckks::cpu

namespace fhelium::ckks::gpu {
decltype(::fhelium::ckks::multiply_two_component_ntt_montgomery)
    multiply_two_component_ntt_montgomery;
decltype(::fhelium::ckks::add_prepared_plaintext_component)
    add_prepared_plaintext_component;
decltype(::fhelium::ckks::add_prepared_plaintext_component_)
    add_prepared_plaintext_component_;
decltype(::fhelium::ckks::add_cyclic_compressed_plaintext_component)
    add_cyclic_compressed_plaintext_component;
decltype(::fhelium::ckks::add_cyclic_compressed_plaintext_component_)
    add_cyclic_compressed_plaintext_component_;
decltype(::fhelium::ckks::add_contiguous_compressed_plaintext_component)
    add_contiguous_compressed_plaintext_component;
decltype(::fhelium::ckks::add_contiguous_compressed_plaintext_component_)
    add_contiguous_compressed_plaintext_component_;
decltype(::fhelium::ckks::add_strided_plaintext_component)
    add_strided_plaintext_component;
decltype(::fhelium::ckks::add_strided_plaintext_component_)
    add_strided_plaintext_component_;
decltype(::fhelium::ckks::rescale_drop_leading_prime_nearest)
    rescale_drop_leading_prime_nearest;
decltype(::fhelium::ckks::rescale_drop_leading_prime_nearest_)
    rescale_drop_leading_prime_nearest_;
decltype(::fhelium::ckks::rescale_drop_leading_prime_truncate)
    rescale_drop_leading_prime_truncate;
decltype(::fhelium::ckks::rescale_drop_leading_prime_truncate_)
    rescale_drop_leading_prime_truncate_;
decltype(::fhelium::ckks::apply_coefficient_galois_automorphism)
    apply_coefficient_galois_automorphism;
decltype(::fhelium::ckks::apply_ntt_galois_automorphism)
    apply_ntt_galois_automorphism;
decltype(::fhelium::ckks::keyswitch_moddown_qp_to_q) keyswitch_moddown_qp_to_q;
decltype(::fhelium::ckks::keyswitch_accumulate_digit_products_)
    keyswitch_accumulate_digit_products_;
decltype(::fhelium::ckks::keyswitch_accumulate_products_)
    keyswitch_accumulate_products_;
}  // namespace fhelium::ckks::gpu
