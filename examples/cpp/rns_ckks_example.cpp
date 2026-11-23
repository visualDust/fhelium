// Calls libfhelium RNS, CKKS, and random functions from C++ and checks every
// result against host modular arithmetic or a published test vector.
//
// Residues use int64 storage with Montgomery radix R = 2^62, and
// fhelium::rns::parameter_table builds the parameter table. The program
// converts standard residues a and b to Montgomery form, multiplies them,
// converts the product back, adds two standard residue Tensors, and evaluates
// the three-component product of two ciphertext-shaped NTT/Montgomery
// payloads. A ChaCha20 key-stream block is checked against RFC 8439,
// Section 2.3.2. It runs on CPU and, when available, on the first CUDA device.

#include <fhelium/fhelium.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

// a * b mod q for q < 2^63 by double-and-add.
uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t q) {
  uint64_t result = 0;
  for (a %= q; b != 0; b >>= 1) {
    if (b & 1) {
      result = (result + a) % q;
    }
    a = (a + a) % q;
  }
  return result;
}

torch::Tensor random_residues(at::IntArrayRef shape,
                              const std::vector<uint64_t>& moduli,
                              std::mt19937_64& rng) {
  auto values = torch::empty(shape, torch::kInt64);
  auto flat =
      values.view({-1, static_cast<int64_t>(moduli.size()), shape.back()});
  auto data = flat.accessor<int64_t, 3>();
  for (int64_t b = 0; b < flat.size(0); ++b) {
    for (int64_t j = 0; j < flat.size(1); ++j) {
      for (int64_t k = 0; k < flat.size(2); ++k) {
        data[b][j][k] = static_cast<int64_t>(rng() % moduli[j]);
      }
    }
  }
  return values;
}

// Reduces lazy residues in [0, 2q_j) to [0, q_j) for comparison.
torch::Tensor reduce(torch::Tensor residues,
                     const std::vector<uint64_t>& moduli) {
  auto q = torch::tensor(std::vector<int64_t>(moduli.begin(), moduli.end()))
               .view({-1, 1});
  return torch::remainder(residues.cpu(), q);
}

bool check(const char* label,
           const torch::Tensor& actual,
           const torch::Tensor& expected) {
  const bool equal = torch::equal(actual, expected);
  std::cout << "  " << label << ": " << (equal ? "ok" : "MISMATCH") << "\n";
  return equal;
}

bool run(const torch::Device& device) {
  std::cout << "device " << device << "\n";
  // Two odd 60-bit moduli and ring dimension N = 16.
  const std::vector<uint64_t> moduli = {1152921504606584833ULL,
                                        1152921504598720513ULL};
  const int64_t n = 16;
  const auto limbs = static_cast<int64_t>(moduli.size());
  std::mt19937_64 rng(7);

  const auto params =
      fhelium::rns::parameter_table(
          std::vector<int64_t>(moduli.begin(), moduli.end()), 4, 1.0,
          torch::kInt64)
          .to(device);
  const auto a_cpu = random_residues({3, limbs, n}, moduli, rng);
  const auto b_cpu = random_residues({3, limbs, n}, moduli, rng);

  auto expected_product = torch::empty_like(a_cpu);
  auto expected_sum = torch::empty_like(a_cpu);
  {
    auto a = a_cpu.accessor<int64_t, 3>();
    auto b = b_cpu.accessor<int64_t, 3>();
    auto p = expected_product.accessor<int64_t, 3>();
    auto s = expected_sum.accessor<int64_t, 3>();
    for (int64_t i = 0; i < a_cpu.size(0); ++i) {
      for (int64_t j = 0; j < limbs; ++j) {
        for (int64_t k = 0; k < n; ++k) {
          p[i][j][k] =
              static_cast<int64_t>(mul_mod(a[i][j][k], b[i][j][k], moduli[j]));
          s[i][j][k] = static_cast<int64_t>(
              (static_cast<uint64_t>(a[i][j][k]) + b[i][j][k]) % moduli[j]);
        }
      }
    }
  }

  bool ok = true;
  const auto a = a_cpu.to(device);
  const auto b = b_cpu.to(device);

  // a * b mod q_j through Montgomery form.
  auto a_montgomery = a.clone();
  auto b_montgomery = b.clone();
  fhelium::rns::to_montgomery_(a_montgomery, params);
  fhelium::rns::to_montgomery_(b_montgomery, params);
  auto product =
      fhelium::rns::montgomery_mul(a_montgomery, b_montgomery, params);
  fhelium::rns::from_montgomery_(product, params);
  ok &= check("rns::montgomery_mul", reduce(product, moduli), expected_product);

  ok &= check("rns::add_standard",
              fhelium::rns::add_standard(a, b, params).cpu(),
              expected_sum);

  // (c0, c1) x (d0, d1) = (c0 d0, c0 d1 + c1 d0, c1 d1) in Montgomery form.
  const auto lhs = a_montgomery.slice(0, 0, 2);
  const auto rhs = b_montgomery.slice(0, 0, 2);
  auto tensor_product =
      fhelium::ckks::multiply_two_component_ntt_montgomery(lhs, rhs, params);
  fhelium::rns::from_montgomery_(tensor_product, params);
  const auto ap = a_cpu.accessor<int64_t, 3>();
  const auto bp = b_cpu.accessor<int64_t, 3>();
  auto expected_tensor_product = torch::empty({3, limbs, n}, torch::kInt64);
  auto e = expected_tensor_product.accessor<int64_t, 3>();
  for (int64_t j = 0; j < limbs; ++j) {
    const uint64_t q = moduli[j];
    for (int64_t k = 0; k < n; ++k) {
      e[0][j][k] = static_cast<int64_t>(mul_mod(ap[0][j][k], bp[0][j][k], q));
      e[1][j][k] = static_cast<int64_t>((mul_mod(ap[0][j][k], bp[1][j][k], q) +
                                         mul_mod(ap[1][j][k], bp[0][j][k], q)) %
                                        q);
      e[2][j][k] = static_cast<int64_t>(mul_mod(ap[1][j][k], bp[1][j][k], q));
    }
  }
  ok &= check("ckks::multiply_two_component_ntt_montgomery",
              reduce(tensor_product, moduli),
              expected_tensor_product);

  // Key bytes 00..1f, block counter 1, and nonce
  // 00:00:00:09:00:00:00:4a:00:00:00:00.
  const auto seed = torch::tensor({int64_t{0x0706050403020100},
                                   int64_t{0x0f0e0d0c0b0a0908},
                                   int64_t{0x1716151413121110},
                                   int64_t{0x1f1e1d1c1b1a1918},
                                   int64_t{0x4a000000}},
                                  torch::kInt64)
                        .to(device);
  const auto block = fhelium::random::chacha20_keystream(
                         seed, 1 | (int64_t{0x09000000} << 32), 1)
                         .to(torch::kInt64)
                         .cpu();
  // clang-format off
  const std::vector<int64_t> rfc8439_block = {
      0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
      0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
      0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
      0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2};
  // clang-format on
  const auto expected_block = torch::tensor(rfc8439_block).view({1, 16});
  ok &= check("random::chacha20_keystream", block, expected_block);
  return ok;
}

}  // namespace

int main() {
  bool ok = run(torch::kCPU);
  if (torch::cuda::is_available()) {
    ok &= run(torch::Device(torch::kCUDA, 0));
  }
  return ok ? 0 : 1;
}
