// Encrypted matrix-vector product with libfhelium from C++.
//
// The program takes the built-in preset slots8192-scale40-depth7, generates
// a secret key, a public key, and the seeded Galois keys of seven rotations,
// keeps them compact (k_0 and the seed rows; the rotations generate k_1), and
// encrypts a vector v of length d = 8 replicated over all slots. It evaluates
// y = M v for a d x d matrix M with the diagonal method of Halevi and Shoup,
//
//   y = sum_{i<d} diag_i * rot(v, -i),   diag_i[j] = M[j mod d][(j + i) mod d],
//
// where rot(x, s) = torch.roll(x, s) and * multiplies slots. The products
// with the encoded diagonals form one NTT-domain weighted sum, which is
// rescaled from depth 0 to depth 1, decrypted, decoded, and compared with
// the product computed in double precision on the host. It also checks
// Montgomery multiplication and the two-component ciphertext product against
// host modular arithmetic. The program runs on CPU and, when available, on
// the first CUDA device, and returns a nonzero status when a check fails.

#include <fhelium/fhelium.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kGenerator = 5;
constexpr int64_t kDimension = 8;
constexpr double kTolerance = 1e-4;
const char* const kPreset = "slots8192-scale40-depth7";
const char* const kNttBackend = "radix2_indexed";

std::vector<int64_t> select(const std::vector<int64_t>& values,
                            const std::vector<int64_t>& ids) {
  std::vector<int64_t> result;
  for (const auto id : ids) {
    result.push_back(values[id]);
  }
  return result;
}

double product(const std::vector<int64_t>& moduli) {
  double result = 1.0;
  for (const auto q : moduli) {
    result *= static_cast<double>(q);
  }
  return result;
}

torch::Tensor positions(const std::vector<int64_t>& ids,
                        const std::vector<int64_t>& basis) {
  std::vector<int64_t> result;
  for (const auto id : ids) {
    result.push_back(std::find(basis.begin(), basis.end(), id) - basis.begin());
  }
  return torch::tensor(result, torch::kInt64);
}

// Builds every table on the host and moves it to the execution device.
class Tables {
 public:
  Tables(const fhelium::ckks::CkksParameters& parameters,
         const torch::Device& device)
      : moduli_(parameters.q_moduli),
        log_n_(parameters.log_n),
        scale_(parameters.scales[0]),
        device_(device) {
    moduli_.insert(
        moduli_.end(), parameters.p_moduli.begin(), parameters.p_moduli.end());
  }

  const std::vector<int64_t>& moduli() const { return moduli_; }

  torch::Tensor parameters(const std::vector<int64_t>& ids) const {
    return fhelium::rns::parameter_table(
               select(moduli_, ids), log_n_, scale_, torch::kInt32)
        .to(device_);
  }

  // The forward and inverse radix-2 indexed NTT tables of the rows `ids`.
  fhelium::RingTables ring(const std::vector<int64_t>& ids) const {
    const auto parameters = this->parameters(ids);
    auto [forward_indices,
          forward_twiddles,
          inverse_indices,
          inverse_twiddles] =
        fhelium::ntt::indexed_radix2_plan(
            select(moduli_, ids), log_n_, torch::kInt32);
    const auto ntt = [&](const torch::Tensor& indices,
                         const torch::Tensor& twiddles) {
      auto montgomery = twiddles.to(device_);
      auto rows = montgomery.view({montgomery.size(0), -1});
      fhelium::rns::to_montgomery_(rows, parameters);
      return fhelium::NttTables{parameters,
                                {montgomery,
                                 indices.select(0, 0).to(device_),
                                 indices.select(0, 1).to(device_)}};
    };
    return {ntt(forward_indices, forward_twiddles),
            ntt(inverse_indices, inverse_twiddles),
            {}};
  }

  // The CRT tables that convert residues over `source` into `target` rows.
  fhelium::CrtTables crt(const std::vector<int64_t>& source,
                         const std::vector<int64_t>& target,
                         const std::tuple<torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor>& tables) const {
    const auto& [matrix,
                 target_reciprocals,
                 normalizers,
                 reciprocals,
                 offsets] = tables;
    return {parameters(source),
            parameters(target),
            to_device(matrix),
            to_device(target_reciprocals),
            to_device(normalizers),
            to_device(reciprocals),
            to_device(offsets)};
  }

  torch::Tensor to_device(const torch::Tensor& tensor) const {
    return tensor.to(device_);
  }

 private:
  std::vector<int64_t> moduli_;
  int64_t log_n_;
  double scale_;
  torch::Device device_;
};

// Advances the nonce of `seed` after a key generation that read `count`
// nonces; encoding and encryption advance the seed they receive.
void advance(torch::Tensor& seed, int64_t count) {
  seed.select(0, 4).add_(count);
}

torch::Tensor encode(const std::vector<std::complex<double>>& slots,
                     const torch::Tensor& pre,
                     const torch::Tensor& twist,
                     torch::Tensor& seed,
                     double scale) {
  auto message =
      torch::empty({static_cast<int64_t>(slots.size())}, torch::kComplexDouble);
  std::copy(
      slots.begin(), slots.end(), message.data_ptr<c10::complex<double>>());
  auto coefficients =
      fhelium::ckks::encode(message.to(seed.device()), pre, twist, seed, scale);
  return coefficients;
}

bool check_matrix_vector(const torch::Device& device) {
  const auto parameters = fhelium::ckks::preset_parameters(kPreset);
  const int64_t n = int64_t{1} << parameters.log_n;
  const int64_t slots = n / 2;
  const auto q_count = static_cast<int64_t>(parameters.q_moduli.size());
  const auto p_count = static_cast<int64_t>(parameters.p_moduli.size());
  const Tables tables(parameters, device);

  // Depth 0 holds every Q prime; the key rows are Q followed by P.
  const auto& q_ids = parameters.q_bases[0];
  std::vector<int64_t> p_ids;
  for (int64_t i = 0; i < p_count; ++i) {
    p_ids.push_back(q_count + i);
  }
  std::vector<int64_t> qp_ids(q_ids);
  qp_ids.insert(qp_ids.end(), p_ids.begin(), p_ids.end());
  const auto q_moduli = select(tables.moduli(), q_ids);
  const auto p_moduli = select(tables.moduli(), p_ids);
  const auto qp_moduli = select(tables.moduli(), qp_ids);

  // Key generation.
  auto seed = fhelium::random::generate_seed().to(device);
  const auto cdt = fhelium::random::gaussian_cdt_table(3.19).to(device);
  auto key_tables = tables.ring(qp_ids);
  key_tables.source_product = tables.to_device(fhelium::rns::product_residues(
      p_moduli, q_moduli, 1, true, torch::kInt32));
  const auto secret_key = fhelium::ckks::create_secret_key(
      n, key_tables, seed, kNttBackend, std::nullopt);
  advance(seed, 1);
  const auto public_key = fhelium::ckks::create_public_key(secret_key,
                                                           key_tables,
                                                           seed,
                                                           cdt,
                                                           kNttBackend,
                                                           std::nullopt,
                                                           std::nullopt);
  advance(seed, 2);
  fhelium::ckks::KeySwitchKeyLayout layout{
      kNttBackend,
      q_ids,
      fhelium::ckks::key_digit_sizes(q_moduli, p_moduli),
      p_ids};
  // Each Galois key is seeded and returned expanded; it consumes one nonce
  // for its seed rows and one per digit error. `compact` keeps k_0 and the
  // seed, halving key memory; the key products generate k_1 from the seed.
  std::vector<fhelium::ckks::KeySwitchKey> galois_keys;
  for (int64_t i = 1; i < kDimension; ++i) {
    galois_keys.push_back(fhelium::ckks::compact_key_switch_key(
        fhelium::ckks::create_galois_key(secret_key,
                                         fhelium::ckks::rotation_galois_element(
                                             parameters.log_n, -i, kGenerator),
                                         key_tables,
                                         seed,
                                         cdt,
                                         layout,
                                         std::nullopt,
                                         std::nullopt)));
    advance(seed, 1 + static_cast<int64_t>(layout.digit_sizes.size()));
  }

  // Encode and encrypt v at depth 0 and the default scale.
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> matrix(kDimension * kDimension);
  std::vector<double> vector(kDimension);
  for (auto& value : matrix) {
    value = uniform(rng);
  }
  for (auto& value : vector) {
    value = uniform(rng);
  }
  auto [pre, post] =
      fhelium::ckks::slot_permutations(parameters.log_n, kGenerator);
  pre = pre.to(device);
  post = post.to(device);
  const auto twist = fhelium::ckks::slot_twist(n, false, device);
  const auto skew = fhelium::ckks::slot_twist(n, true, device);
  std::vector<std::complex<double>> replicated(slots);
  for (int64_t j = 0; j < slots; ++j) {
    replicated[j] = vector[j % kDimension];
  }
  const double input_scale = parameters.scales[0];
  const auto q_tables = tables.ring(q_ids);
  auto ciphertext =
      fhelium::ckks::encrypt(encode(replicated, pre, twist, seed, input_scale),
                             public_key,
                             q_tables,
                             seed,
                             cdt,
                             kNttBackend,
                             0,
                             "coefficient");

  // Rotations: hybrid key switching with normalized CRT ModUp and an
  // approximate ModDown that divides by P with the centered CRT remainder.
  fhelium::KeySwitchTables rotation_tables;
  const auto q_ring = tables.ring(q_ids);
  const auto qp_ring = tables.ring(qp_ids);
  rotation_tables.q_parameters = q_ring.forward.parameters;
  rotation_tables.qp_parameters = qp_ring.forward.parameters;
  rotation_tables.q_forward = q_ring.forward;
  rotation_tables.q_inverse = q_ring.inverse;
  rotation_tables.qp_forward = qp_ring.forward;
  rotation_tables.qp_inverse = qp_ring.inverse;
  rotation_tables.moddown_crt = tables.crt(
      p_ids,
      q_ids,
      fhelium::rns::crt_quotient_tables(p_moduli, q_moduli, {}, torch::kInt32));
  rotation_tables.p_inverse = tables.to_device(fhelium::rns::product_residues(
      p_moduli, q_moduli, -1, true, torch::kInt32));
  fhelium::ckks::KeySwitchOptions options;
  options.ntt_backend = kNttBackend;
  options.basis_conversion = "normalized_crt";
  options.p_count = p_count;
  rotation_tables.digits.resize(layout.digit_sizes.size());
  int64_t start = 0;
  for (size_t digit = 0; digit < layout.digit_sizes.size(); ++digit) {
    const int64_t stop = start + layout.digit_sizes[digit];
    options.digits.insert(options.digits.end(),
                          {start, stop, static_cast<int64_t>(digit)});
    const std::vector<int64_t> digit_ids(q_ids.begin() + start,
                                         q_ids.begin() + stop);
    rotation_tables.digits[digit].crt =
        tables.crt(digit_ids,
                   qp_ids,
                   fhelium::rns::crt_basis_extension_tables(
                       select(tables.moduli(), digit_ids),
                       qp_moduli,
                       true,
                       torch::kInt32));
    start = stop;
  }

  // Diagonal plaintexts at the scale that makes the rescaled product carry
  // the depth-1 default scale: input * plaintext * R / D = scales[1].
  const auto& target_ids = parameters.q_bases[1];
  std::vector<int64_t> dropped_ids;
  std::vector<int64_t> kept_ids;
  std::vector<int64_t> restored_ids;
  for (const auto id : q_ids) {
    const bool kept =
        std::find(target_ids.begin(), target_ids.end(), id) != target_ids.end();
    (kept ? kept_ids : dropped_ids).push_back(id);
  }
  for (const auto id : target_ids) {
    if (std::find(q_ids.begin(), q_ids.end(), id) == q_ids.end()) {
      restored_ids.push_back(id);
    }
  }
  const double ratio = product(select(tables.moduli(), dropped_ids)) /
                       product(select(tables.moduli(), restored_ids));
  const double plaintext_scale = parameters.scales[1] * ratio / input_scale;
  const auto& q_parameters = rotation_tables.q_parameters;
  // Rotation by -i applies the automorphism of Galois element g^i.
  const auto galois = [&](int64_t step) {
    const auto slots =
        fhelium::ckks::galois_table_set(parameters.log_n,
                                        fhelium::ckks::rotation_galois_element(
                                            parameters.log_n, step, kGenerator),
                                        device);
    return fhelium::GaloisTables{slots[0], slots[1], slots[2]};
  };

  std::vector<torch::Tensor> rotated;
  std::vector<torch::Tensor> diagonals;
  for (int64_t i = 0; i < kDimension; ++i) {
    const auto term = i == 0 ? ciphertext
                             : fhelium::ckks::rotate(ciphertext,
                                                     galois_keys[i - 1],
                                                     galois(-i),
                                                     rotation_tables,
                                                     options);
    rotated.push_back(fhelium::ntt::transform(
        term, rotation_tables.q_forward, kNttBackend, "forward_to_montgomery"));
    std::vector<std::complex<double>> diagonal(slots);
    for (int64_t j = 0; j < slots; ++j) {
      const int64_t row = j % kDimension;
      diagonal[j] = matrix[row * kDimension + (row + i) % kDimension];
    }
    auto residues = fhelium::rns::lift_integer_coefficients(
        encode(diagonal, pre, twist, seed, plaintext_scale),
        q_parameters.select(0, 0));
    fhelium::rns::reduce_to_standard_(residues, q_parameters);
    diagonals.push_back(fhelium::ntt::transform(residues,
                                                rotation_tables.q_forward,
                                                kNttBackend,
                                                "forward_to_montgomery"));
  }
  auto sum =
      fhelium::rns::montgomery_weighted_sum(rotated, diagonals, q_parameters);
  fhelium::ntt::transform_(
      sum, rotation_tables.q_inverse, kNttBackend, "inverse_to_standard");

  // Rescale from the depth-0 basis B to the depth-1 basis B'.
  std::vector<int64_t> ordered_ids(target_ids);
  ordered_ids.insert(ordered_ids.end(), dropped_ids.begin(), dropped_ids.end());
  const auto dropped_moduli = select(tables.moduli(), dropped_ids);
  const auto offset = [&](const std::vector<int64_t>& ids) {
    return tables.to_device(
        fhelium::rns::quotient_offset_residues({},
                                               dropped_moduli,
                                               false,
                                               true,
                                               select(tables.moduli(), ids),
                                               torch::kInt32)
            .view({-1, 1})
            .expand({-1, n})
            .contiguous());
  };
  fhelium::RescaleTables rescale_tables;
  rescale_tables.source_parameters = tables.parameters(q_ids);
  rescale_tables.target_parameters = tables.parameters(target_ids);
  rescale_tables.dropped_parameters = tables.parameters(dropped_ids);
  rescale_tables.ordered_parameters = tables.parameters(ordered_ids);
  rescale_tables.drop_positions =
      tables.to_device(positions(dropped_ids, q_ids));
  rescale_tables.keep_positions = tables.to_device(positions(kept_ids, q_ids));
  rescale_tables.inverses =
      tables.to_device(fhelium::rns::trailing_quotient_inverses(
          select(tables.moduli(), ordered_ids),
          static_cast<int64_t>(dropped_ids.size()),
          torch::kInt32));
  rescale_tables.dropped_offset = offset(dropped_ids);
  rescale_tables.target_offset = offset(target_ids);
  if (!restored_ids.empty()) {
    rescale_tables.target_keep_positions =
        tables.to_device(positions(kept_ids, target_ids));
    rescale_tables.restore = tables.to_device(
        fhelium::rns::product_residues(select(tables.moduli(), restored_ids),
                                       q_moduli,
                                       1,
                                       true,
                                       torch::kInt32));
  }
  const auto result = fhelium::rns::rational_rescale(
      sum, rescale_tables, "coefficient", kNttBackend);

  // The expanded form of a key gives the same rotation bitwise.
  const bool forms_match =
      fhelium::ckks::rotate(
          ciphertext,
          fhelium::ckks::expand_key_switch_key(galois_keys[0]),
          galois(-1),
          rotation_tables,
          options)
          .equal(fhelium::ckks::rotate(ciphertext,
                                       galois_keys[0],
                                       galois(-1),
                                       rotation_tables,
                                       options));

  // Decrypt at depth 1 and decode at the depth-1 scale.
  const auto target_moduli = select(tables.moduli(), target_ids);
  fhelium::DecryptionTables decrypt_tables;
  decrypt_tables.ring = tables.ring(target_ids);
  auto [normalizers, propagation, extension] = fhelium::rns::mixed_radix_tables(
      target_moduli, target_moduli, torch::kInt32);
  decrypt_tables.reconstruction_parameters = tables.parameters(target_ids);
  decrypt_tables.normalizers = tables.to_device(normalizers);
  decrypt_tables.propagation = tables.to_device(propagation);
  decrypt_tables.half_digits = tables.to_device(
      fhelium::rns::half_product_digits(target_moduli, torch::kInt32));
  const auto secret_rows = secret_key.index_select(
      0, torch::tensor(target_ids, torch::kInt64).to(device));
  const auto coefficients = fhelium::ckks::decrypt(
      result, secret_rows, decrypt_tables, kNttBackend, 0, "coefficient");
  const auto decoded =
      fhelium::ckks::decode(
          coefficients, post, skew, parameters.scales[1], false)
          .cpu();

  double max_error = 0.0;
  const auto values = decoded.accessor<c10::complex<double>, 1>();
  for (int64_t j = 0; j < slots; ++j) {
    const int64_t row = j % kDimension;
    double expected = 0.0;
    for (int64_t k = 0; k < kDimension; ++k) {
      expected += matrix[row * kDimension + k] * vector[k];
    }
    max_error = std::max(max_error,
                         std::abs(std::complex<double>(values[j]) - expected));
  }
  std::cout << "  " << kDimension << "x" << kDimension
            << " matrix-vector product over " << slots
            << " slots, depth 0 -> 1: max error " << max_error << "\n"
            << "  compact and expanded key rotations: "
            << (forms_match ? "match" : "MISMATCH") << "\n";
  return max_error < kTolerance && forms_match;
}

// a * b mod q by double-and-add.
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

// Checks Montgomery multiplication and the two-component ciphertext product
// on random residues modulo two 60-bit primes against host arithmetic.
bool check_primitives(const torch::Device& device) {
  const std::vector<int64_t> moduli = {1152921504606584833LL,
                                       1152921504598720513LL};
  const int64_t n = 16;
  const auto limbs = static_cast<int64_t>(moduli.size());
  const auto params =
      fhelium::rns::parameter_table(moduli, 4, 1.0, torch::kInt64).to(device);
  std::mt19937_64 rng(11);
  auto a_cpu = torch::empty({2, limbs, n}, torch::kInt64);
  auto b_cpu = torch::empty({2, limbs, n}, torch::kInt64);
  auto a = a_cpu.accessor<int64_t, 3>();
  auto b = b_cpu.accessor<int64_t, 3>();
  for (int64_t c = 0; c < 2; ++c) {
    for (int64_t j = 0; j < limbs; ++j) {
      for (int64_t k = 0; k < n; ++k) {
        a[c][j][k] = static_cast<int64_t>(rng() % moduli[j]);
        b[c][j][k] = static_cast<int64_t>(rng() % moduli[j]);
      }
    }
  }
  auto lhs = a_cpu.to(device, false, true);
  auto rhs = b_cpu.to(device, false, true);
  fhelium::rns::to_montgomery_(lhs, params);
  fhelium::rns::to_montgomery_(rhs, params);
  auto square = fhelium::rns::montgomery_mul(lhs, rhs, params);
  auto product3 =
      fhelium::ckks::multiply_two_component_ntt_montgomery(lhs, rhs, params);
  fhelium::rns::from_montgomery_(square, params);
  fhelium::rns::from_montgomery_(product3, params);
  const auto q = torch::tensor(moduli).view({-1, 1});
  square = torch::remainder(square.cpu(), q);
  product3 = torch::remainder(product3.cpu(), q);
  auto s = square.accessor<int64_t, 3>();
  auto p = product3.accessor<int64_t, 3>();
  bool ok = true;
  for (int64_t j = 0; j < limbs; ++j) {
    const auto m = static_cast<uint64_t>(moduli[j]);
    for (int64_t k = 0; k < n; ++k) {
      const auto product = [&](int64_t x, int64_t y) {
        return mul_mod(a[x][j][k], b[y][j][k], m);
      };
      ok &= static_cast<uint64_t>(s[0][j][k]) == product(0, 0);
      ok &= static_cast<uint64_t>(s[1][j][k]) == product(1, 1);
      ok &= static_cast<uint64_t>(p[0][j][k]) == product(0, 0);
      ok &= static_cast<uint64_t>(p[1][j][k]) ==
            (product(0, 1) + product(1, 0)) % m;
      ok &= static_cast<uint64_t>(p[2][j][k]) == product(1, 1);
    }
  }
  std::cout << "  Montgomery and ciphertext products: "
            << (ok ? "match" : "MISMATCH") << "\n";
  return ok;
}

bool run(const torch::Device& device) {
  std::cout << "device " << device << "\n";
  const bool primitives = check_primitives(device);
  const bool matrix_vector = check_matrix_vector(device);
  return primitives && matrix_vector;
}

}  // namespace

int main() {
  bool ok = run(torch::kCPU);
  if (torch::cuda::is_available()) {
    ok &= run(torch::Device(torch::kCUDA, 0));
  }
  return ok ? 0 : 1;
}
