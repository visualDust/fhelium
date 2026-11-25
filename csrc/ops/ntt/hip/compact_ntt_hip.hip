#include "compact_ntt_hip.h"

#include <ATen/MemoryOverlap.h>
#include <c10/hip/HIPStream.h>

#include "../../common/key_rows.h"
#include "../../common/rns_batch.h"
#include "../ntt_validation.h"
#include "compact_ntt_phases.hpp"

namespace compact = fhelium::ntt::compact;

namespace {

template <typename Function>
void dispatch(const torch::Tensor& tensor, const char* name, Function&& function) {
  hipSetDevice(tensor.device().index());
  const auto stream = c10::hip::getCurrentHIPStream(tensor.device().index());
  if (tensor.scalar_type() == at::kInt) {
    function(int32_t{}, stream.stream());
  } else {
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " requires int32 or int64 residues");
    function(int64_t{}, stream.stream());
  }
}

}  // namespace

void compact_forward_ntt_hip(const torch::Tensor& source,
                              torch::Tensor destination,
                              const torch::Tensor& forward_twiddles,
                              const torch::Tensor& rns_params,
                              CompactForwardInput input) {
  const auto in = view_rns_batch_3d(source, "residues");
  const auto out = view_rns_batch_3d(destination, "out");
  dispatch(in, "compact_forward_ntt_hip", [&](auto zero, hipStream_t stream) {
    using T = decltype(zero);
    const auto s = compact::operand<T>(in);
    const auto d = compact::operand<T>(out);
    if (input == CompactForwardInput::Standard) {
      compact::forward<T, compact::Load::ToMontgomery>(
          s, d, compact::StoreValues{}, forward_twiddles, rns_params, stream);
    } else {
      compact::forward<T, compact::Load::Plain>(
          s, d, compact::StoreValues{}, forward_twiddles, rns_params, stream);
    }
  });
}

void compact_inverse_ntt_hip(torch::Tensor residues,
                              const torch::Tensor& inverse_twiddles,
                              const torch::Tensor& rns_params,
                              CompactInverseResult result) {
  const auto values = view_rns_batch_3d(residues, "residues");
  dispatch(values, "compact_inverse_ntt_hip", [&](auto zero, hipStream_t stream) {
    using T = decltype(zero);
    const auto v = compact::operand<T>(values);
    auto run = [&](auto output) {
      output.data = v.data;
      output.batch_stride = v.layout.batch_stride;
      output.row_stride = v.layout.row_stride;
      compact::inverse<T>(v, output, inverse_twiddles, rns_params, stream);
    };
    using compact::InverseOutput;
    switch (result) {
      case CompactInverseResult::Montgomery:
        run(compact::StoreInverse<T, InverseOutput::Montgomery>{});
        break;
      case CompactInverseResult::StandardLazy:
        run(compact::StoreInverse<T, InverseOutput::StandardLazy>{});
        break;
      case CompactInverseResult::Standard:
        run(compact::StoreInverse<T, InverseOutput::Standard>{});
        break;
      case CompactInverseResult::Centered:
        run(compact::StoreInverse<T, InverseOutput::Centered>{});
        break;
    }
  });
}

void compact_forward_ntt_add_scaled_hip(torch::Tensor source,
                                         const torch::Tensor& addend,
                                         const torch::Tensor& multiplier,
                                         const torch::Tensor& forward_twiddles,
                                         const torch::Tensor& rns_params) {
  TORCH_CHECK(addend.device() == source.device() &&
                  addend.scalar_type() == source.scalar_type() &&
                  addend.sizes() == source.sizes(),
              "NTT addend must match source shape, dtype and device");
  TORCH_CHECK(multiplier.device() == source.device() &&
                  multiplier.scalar_type() == source.scalar_type() &&
                  multiplier.dim() == 1 && multiplier.size(0) == source.size(-2),
              "NTT multiplier must match prime rows, dtype and device");
  fhelium::ntt::validate_no_residue_overlap(source, addend, "NTT addend");
  fhelium::ntt::validate_no_residue_overlap(source, multiplier, "NTT multiplier");
  const auto values = view_rns_batch_3d(source, "source");
  const auto addends = view_rns_batch_3d(addend, "addend");
  TORCH_CHECK(addends.stride(2) == 1, "NTT addend requires unit coefficient stride");
  dispatch(values, "compact_forward_ntt_add_scaled_hip", [&](auto zero, hipStream_t stream) {
    using T = decltype(zero);
    const auto v = compact::operand<T>(values);
    const compact::StoreAddScaled<T> output{
        v.data,          v.layout.batch_stride,  v.layout.row_stride,
        addends.data_ptr<T>(), addends.stride(0), addends.stride(1),
        multiplier.data_ptr<T>(), multiplier.stride(0)};
    compact::forward<T, compact::Load::ToMontgomery>(
        v, v, output, forward_twiddles, rns_params, stream);
  });
}

void compact_forward_ntt_keyswitch_accumulate_hip(
    torch::Tensor coefficient_digit_qp,
    const torch::Tensor& forward_twiddles,
    const torch::Tensor& rns_params,
    const torch::Tensor& key_digit_qp,
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    int64_t key_row_start,
    int64_t key_p_row_start) {
  for (const auto& tensor : {key_digit_qp, accumulator0_qp, accumulator1_qp}) {
    TORCH_CHECK(tensor.device() == coefficient_digit_qp.device() &&
                    tensor.scalar_type() == coefficient_digit_qp.scalar_type(),
                "Fused NTT key products require one dtype and HIP device");
  }
  for (const auto& tensor : {accumulator0_qp, accumulator1_qp}) {
    at::assert_no_internal_overlap(tensor);
    for (const auto& input :
         {coefficient_digit_qp, key_digit_qp, forward_twiddles, rns_params}) {
      at::assert_no_overlap(tensor, input);
    }
  }
  at::assert_no_overlap(accumulator0_qp, accumulator1_qp);
  at::assert_no_overlap(coefficient_digit_qp, key_digit_qp);
  const auto digit = view_rns_batch_3d(coefficient_digit_qp, "coefficient_digit_qp");
  const auto accumulator0 = view_rns_batch_3d(accumulator0_qp, "accumulator0_qp");
  const auto accumulator1 = view_rns_batch_3d(accumulator1_qp, "accumulator1_qp");
  constexpr const char* operation = "forward_ntt_compact_keyswitch_accumulate";
  check_rns_binary_3d(accumulator0, digit, operation, false);
  check_rns_binary_3d(accumulator1, digit, operation, false);
  TORCH_CHECK(key_digit_qp.dim() == 3 && key_digit_qp.size(0) == 2 &&
                  key_digit_qp.size(2) == digit.size(2),
              "key_digit_qp must have [2, QP row, coefficient] layout");
  TORCH_CHECK(key_digit_qp.stride(2) == 1 && accumulator0.stride(2) == 1 &&
                  accumulator1.stride(2) == 1,
              "Fused NTT key products require unit coefficient strides");
  for (const auto& tensor : {key_digit_qp, accumulator0, accumulator1}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0 &&
                    tensor.stride(0) * tensor.element_size() % 16 == 0 &&
                    tensor.stride(1) * tensor.element_size() % 16 == 0,
                "Fused NTT key products require 16-byte aligned rows");
  }
  const auto key_rows = fhelium::detail::key_row_map(
      digit.size(1), key_digit_qp.size(1), key_row_start, key_p_row_start, operation);
  dispatch(digit, operation, [&](auto zero, hipStream_t stream) {
    using T = decltype(zero);
    const auto d = compact::operand<T>(digit);
    const compact::AccumulateKeyProducts<T> output{
        key_digit_qp.data_ptr<T>(),
        key_digit_qp.stride(0),
        key_digit_qp.stride(1),
        accumulator0.data_ptr<T>(),
        {accumulator0.stride(0), accumulator0.stride(1)},
        accumulator1.data_ptr<T>(),
        {accumulator1.stride(0), accumulator1.stride(1)},
        static_cast<int>(key_row_start),
        static_cast<int>(key_rows.q_rows),
        static_cast<int>(key_rows.p_row_start)};
    compact::forward<T, compact::Load::Plain>(
        d, d, output, forward_twiddles, rns_params, stream);
  });
}
