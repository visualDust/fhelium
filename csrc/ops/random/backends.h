#pragma once

#include <fhelium/random.h>

// Backend entry points behind the public functions of the same name.
// The cpu namespace is compiled from cpu/ sources; the gpu namespace is
// compiled from either cuda/ or hip/ sources.

namespace fhelium::random::cpu {
decltype(::fhelium::random::chacha20_keystream) chacha20_keystream;
decltype(::fhelium::random::sample_uniform_residues) sample_uniform_residues;
decltype(::fhelium::random::sample_discrete_gaussian) sample_discrete_gaussian;
decltype(::fhelium::random::sample_centered_binomial) sample_centered_binomial;
decltype(::fhelium::random::sample_ternary) sample_ternary;
decltype(::fhelium::random::add_discrete_gaussian_) add_discrete_gaussian_;
decltype(::fhelium::random::stochastic_round) stochastic_round;
decltype(::fhelium::random::expand_seeded_key) expand_seeded_key;
}  // namespace fhelium::random::cpu

namespace fhelium::random::gpu {
decltype(::fhelium::random::chacha20_keystream) chacha20_keystream;
decltype(::fhelium::random::sample_uniform_residues) sample_uniform_residues;
decltype(::fhelium::random::sample_discrete_gaussian) sample_discrete_gaussian;
decltype(::fhelium::random::sample_centered_binomial) sample_centered_binomial;
decltype(::fhelium::random::sample_ternary) sample_ternary;
decltype(::fhelium::random::add_discrete_gaussian_) add_discrete_gaussian_;
decltype(::fhelium::random::stochastic_round) stochastic_round;
decltype(::fhelium::random::expand_seeded_key) expand_seeded_key;
}  // namespace fhelium::random::gpu
