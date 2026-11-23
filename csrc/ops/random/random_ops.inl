// Argument handling shared by the CPU and GPU samplers. Included inside the
// backend namespace after the backend defines:
//
//   template <typename Sampler, typename Output>
//   void run(const at::Tensor& seed, const Sampler&, const Output&,
//            int64_t count);
//   void keystream_into(const at::Tensor& seed, uint64_t first_block,
//                       int64_t block_count, uint32_t* out);
//
// `run` evaluates every stream block holding words of samples [0, count).

namespace detail_ops {

using ::fhelium::random::detail::AddResidueOutput;
using ::fhelium::random::detail::CenteredBinomialSampler;
using ::fhelium::random::detail::DirectOutput;
using ::fhelium::random::detail::GaussianSampler;
using ::fhelium::random::detail::ResidueOutput;
using ::fhelium::random::detail::StochasticRoundSampler;
using ::fhelium::random::detail::StridedOutput;
using ::fhelium::random::detail::TernarySampler;
using ::fhelium::random::detail::UniformResidueSampler;

inline void check_seed(const at::Tensor& seed, const char* operation) {
  TORCH_CHECK(seed.scalar_type() == at::kLong && seed.dim() == 1 &&
                  seed.size(0) == ::fhelium::random::detail::kSeedWords &&
                  seed.is_contiguous(),
              operation,
              " requires seed to be a contiguous int64 Tensor of shape [5]");
}

inline void check_peer(const at::Tensor& seed,
                       const at::Tensor& tensor,
                       const char* operation,
                       const char* name) {
  TORCH_CHECK(tensor.device() == seed.device(),
              operation,
              " requires ",
              name,
              " on the device of seed");
}

inline void check_cdt_table(const at::Tensor& seed,
                            const at::Tensor& cdt_table,
                            const char* operation) {
  check_peer(seed, cdt_table, operation, "cdt_table");
  TORCH_CHECK(cdt_table.scalar_type() == at::kLong && cdt_table.dim() == 2 &&
                  cdt_table.size(1) == 2 && cdt_table.is_contiguous(),
              operation,
              " requires a contiguous int64 cdt_table of shape [rows, 2]");
}

inline int64_t coefficient_count(at::IntArrayRef shape, const char* operation) {
  TORCH_CHECK(!shape.empty(),
              operation,
              " requires a shape ending in the coefficient axis");
  int64_t count = 1;
  for (const int64_t extent : shape) {
    TORCH_CHECK(extent >= 0, operation, " requires nonnegative extents");
    count *= extent;
  }
  return count;
}

inline void check_rns_params(const at::Tensor& seed,
                             const at::Tensor& rns_params,
                             const char* operation) {
  check_peer(seed, rns_params, operation, "rns_params");
  TORCH_CHECK((rns_params.scalar_type() == at::kInt ||
               rns_params.scalar_type() == at::kLong) &&
                  rns_params.dim() == 2 && rns_params.size(0) >= 1,
              operation,
              " requires int32 or int64 rns_params of shape [rows, limb]");
}

inline std::vector<int64_t> residue_shape(at::IntArrayRef shape,
                                          int64_t limbs) {
  std::vector<int64_t> result(shape.begin(), shape.end() - 1);
  result.push_back(limbs);
  result.push_back(shape.back());
  return result;
}

// Writes integer samples as centered int64 values or as reduced residues.
// Untyped data pointers keep this template parseable by nvcc's MSVC-mode
// front end, which rejects member-template calls here.
template <typename Sampler>
at::Tensor sample_integers(const at::Tensor& seed,
                           at::IntArrayRef shape,
                           const Sampler& sampler,
                           const std::optional<at::Tensor>& rns_params,
                           const char* operation) {
  const int64_t count = coefficient_count(shape, operation);
  if (!rns_params.has_value()) {
    auto out = at::empty(shape, seed.options());
    run(seed,
        sampler,
        detail_ops::DirectOutput<int64_t>{
            static_cast<int64_t*>(out.data_ptr())},
        count);
    return out;
  }
  check_rns_params(seed, *rns_params, operation);
  const auto twice_modulus = rns_params->select(0, 0);
  auto out = at::empty(residue_shape(shape, twice_modulus.size(0)),
                       rns_params->options());
  AT_DISPATCH_INDEX_TYPES(out.scalar_type(), "fhelium_random_sample", [&] {
    run(seed,
        sampler,
        detail_ops::ResidueOutput<index_t>{
            static_cast<index_t*>(out.data_ptr()),
            static_cast<const index_t*>(twice_modulus.const_data_ptr()),
            twice_modulus.stride(0),
            twice_modulus.size(0),
            shape.back()},
        count);
  });
  return out;
}

}  // namespace detail_ops

at::Tensor chacha20_keystream(const at::Tensor& seed,
                              int64_t first_block,
                              int64_t block_count) {
  constexpr const char* operation = "chacha20_keystream";
  detail_ops::check_seed(seed, operation);
  TORCH_CHECK(block_count >= 0, operation, " requires block_count >= 0");
  auto out = at::empty({block_count, ::fhelium::random::detail::kBlockWords},
                       seed.options().dtype(at::kUInt32));
  keystream_into(seed,
                 static_cast<uint64_t>(first_block),
                 block_count,
                 out.mutable_data_ptr<uint32_t>());
  return out;
}

at::Tensor sample_uniform_residues(const at::Tensor& seed,
                                   at::IntArrayRef shape,
                                   const at::Tensor& rns_params) {
  constexpr const char* operation = "sample_uniform_residues";
  detail_ops::check_seed(seed, operation);
  detail_ops::check_rns_params(seed, rns_params, operation);
  const int64_t count = detail_ops::coefficient_count(shape, operation);
  const auto twice_modulus = rns_params.select(0, 0);
  const int64_t limbs = twice_modulus.size(0);
  auto out =
      at::empty(detail_ops::residue_shape(shape, limbs), rns_params.options());
  AT_DISPATCH_INDEX_TYPES(out.scalar_type(), operation, [&] {
    run(seed,
        detail_ops::UniformResidueSampler<index_t>{
            twice_modulus.const_data_ptr<index_t>(),
            twice_modulus.stride(0),
            limbs,
            shape.back()},
        detail_ops::DirectOutput<index_t>{out.data_ptr<index_t>()},
        count * limbs);
  });
  return out;
}

at::Tensor sample_discrete_gaussian(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    const at::Tensor& cdt_table,
    const std::optional<at::Tensor>& rns_params) {
  constexpr const char* operation = "sample_discrete_gaussian";
  detail_ops::check_seed(seed, operation);
  detail_ops::check_cdt_table(seed, cdt_table, operation);
  return detail_ops::sample_integers(
      seed,
      shape,
      detail_ops::GaussianSampler{cdt_table.const_data_ptr<int64_t>(),
                                  cdt_table.size(0)},
      rns_params,
      operation);
}

at::Tensor sample_centered_binomial(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    int64_t k,
    const std::optional<at::Tensor>& rns_params) {
  constexpr const char* operation = "sample_centered_binomial";
  detail_ops::check_seed(seed, operation);
  TORCH_CHECK(k >= 1 && k <= 32, operation, " requires 1 <= k <= 32");
  const uint32_t mask =
      k == 32 ? 0xffffffffu : ((uint32_t{1} << static_cast<int>(k)) - 1u);
  return detail_ops::sample_integers(seed,
                                     shape,
                                     detail_ops::CenteredBinomialSampler{mask},
                                     rns_params,
                                     operation);
}

at::Tensor sample_ternary(const at::Tensor& seed,
                          at::IntArrayRef shape,
                          double nonzero_probability,
                          const std::optional<at::Tensor>& rns_params) {
  constexpr const char* operation = "sample_ternary";
  detail_ops::check_seed(seed, operation);
  TORCH_CHECK(nonzero_probability >= 0.0 && nonzero_probability <= 1.0,
              operation,
              " requires 0 <= nonzero_probability <= 1");
  // floor(rho * 2^63); rho = 1 maps to 2^63, which every 63-bit value is below.
  const auto threshold =
      static_cast<uint64_t>(std::ldexp(nonzero_probability, 63));
  return detail_ops::sample_integers(seed,
                                     shape,
                                     detail_ops::TernarySampler{threshold},
                                     rns_params,
                                     operation);
}

void add_discrete_gaussian_(at::Tensor& residues,
                            const at::Tensor& seed,
                            const at::Tensor& cdt_table,
                            const at::Tensor& rns_params) {
  constexpr const char* operation = "add_discrete_gaussian_";
  detail_ops::check_seed(seed, operation);
  detail_ops::check_cdt_table(seed, cdt_table, operation);
  detail_ops::check_rns_params(seed, rns_params, operation);
  detail_ops::check_peer(seed, residues, operation, "residues");
  const auto twice_modulus = rns_params.select(0, 0);
  TORCH_CHECK(residues.scalar_type() == rns_params.scalar_type() &&
                  residues.dim() >= 2 &&
                  residues.size(-2) == twice_modulus.size(0) &&
                  residues.is_contiguous(),
              operation,
              " requires contiguous residues of shape [*batch, limb, "
              "coefficient] with the dtype and limbs of rns_params");
  const int64_t coefficients = residues.size(-1);
  const int64_t count =
      residues.numel() / std::max<int64_t>(residues.size(-2), 1);
  AT_DISPATCH_INDEX_TYPES(residues.scalar_type(), operation, [&] {
    run(seed,
        detail_ops::GaussianSampler{cdt_table.const_data_ptr<int64_t>(),
                                    cdt_table.size(0)},
        detail_ops::AddResidueOutput<index_t>{
            residues.data_ptr<index_t>(),
            twice_modulus.const_data_ptr<index_t>(),
            twice_modulus.stride(0),
            twice_modulus.size(0),
            coefficients},
        count);
  });
}

at::Tensor stochastic_round(const at::Tensor& seed,
                            const at::Tensor& values,
                            int64_t word_stride) {
  constexpr const char* operation = "stochastic_round";
  detail_ops::check_seed(seed, operation);
  detail_ops::check_peer(seed, values, operation, "values");
  TORCH_CHECK(values.scalar_type() == at::kDouble,
              operation,
              " requires float64 values");
  TORCH_CHECK(word_stride >= 1, operation, " requires word_stride >= 1");
  const auto source = values.contiguous();
  auto out = at::empty(values.sizes(), seed.options());
  const detail_ops::StochasticRoundSampler sampler{
      source.const_data_ptr<double>(), word_stride};
  if (word_stride == 1) {
    run(seed,
        sampler,
        detail_ops::DirectOutput<int64_t>{out.data_ptr<int64_t>()},
        source.numel());
  } else {
    run(seed,
        sampler,
        detail_ops::StridedOutput<int64_t>{out.data_ptr<int64_t>(),
                                           word_stride},
        source.numel() * word_stride);
  }
  return out;
}
