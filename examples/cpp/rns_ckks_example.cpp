// Calls libfhelium RNS and CKKS functions from C++ and checks every result
// against 128-bit integer arithmetic.
//
// Residues use int64 storage with Montgomery radix R = 2^62. The program
// converts standard residues a and b to Montgomery form, multiplies them,
// converts the product back, adds two standard residue Tensors, and evaluates
// the three-component product of two ciphertext-shaped NTT/Montgomery
// payloads. It runs on CPU and, when available, on the first CUDA device.

#include <fhelium/fhelium.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using u128 = unsigned __int128;

constexpr int kRadixBits = 62;
constexpr int kHalfRadixBits = kRadixBits / 2;
constexpr uint64_t kRadixMask = (uint64_t{1} << kRadixBits) - 1;
constexpr uint64_t kHalfMask = (uint64_t{1} << kHalfRadixBits) - 1;

uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t q) {
  return static_cast<uint64_t>(static_cast<u128>(a) * b % q);
}

uint64_t pow_mod(uint64_t base, uint64_t exponent, uint64_t q) {
  uint64_t result = 1 % q;
  for (base %= q; exponent != 0; exponent >>= 1) {
    if (exponent & 1) {
      result = mul_mod(result, base, q);
    }
    base = mul_mod(base, base, q);
  }
  return result;
}

// -q^{-1} mod 2^62 by Newton iteration on odd q.
uint64_t neg_inverse_mod_radix(uint64_t q) {
  uint64_t inverse = q;
  for (int i = 0; i < 6; ++i) {
    inverse *= 2 - q * inverse;
  }
  return (0 - inverse) & kRadixMask;
}

// [RNS_PARAMETER_COUNT, limb] table for odd moduli q_j < 2^60.
torch::Tensor rns_parameters(const std::vector<uint64_t>& moduli,
                             int64_t ring_dimension) {
  using namespace fhelium::rns;
  const auto limbs = static_cast<int64_t>(moduli.size());
  auto table = torch::zeros({RNS_PARAMETER_COUNT, limbs}, torch::kInt64);
  auto rows = table.accessor<int64_t, 2>();
  for (int64_t j = 0; j < limbs; ++j) {
    const uint64_t q = moduli[j];
    const uint64_t r_mod_q = static_cast<uint64_t>((u128{1} << kRadixBits) % q);
    const uint64_t neg_inverse = neg_inverse_mod_radix(q);
    const uint64_t n_inverse = pow_mod(ring_dimension % q, q - 2, q);
    rows[RNS_PARAM_TWICE_MODULUS][j] = static_cast<int64_t>(2 * q);
    rows[RNS_PARAM_MODULUS_LO][j] = static_cast<int64_t>(q & kHalfMask);
    rows[RNS_PARAM_MODULUS_HI][j] = static_cast<int64_t>(q >> kHalfRadixBits);
    rows[RNS_PARAM_NEG_INV_MODULUS_LO][j] =
        static_cast<int64_t>(neg_inverse & kHalfMask);
    rows[RNS_PARAM_NEG_INV_MODULUS_HI][j] =
        static_cast<int64_t>(neg_inverse >> kHalfRadixBits);
    rows[RNS_PARAM_R2][j] = static_cast<int64_t>(mul_mod(r_mod_q, r_mod_q, q));
    rows[RNS_PARAM_SCALED_R2][j] = rows[RNS_PARAM_R2][j];
    rows[RNS_PARAM_N_INV_MONTGOMERY][j] =
        static_cast<int64_t>(mul_mod(n_inverse, r_mod_q, q));
  }
  return table;
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

  const auto params = rns_parameters(moduli, n).to(device);
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
