#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "tensor_validation.h"

namespace {

using fhelium::ckks::cpu::check_cpu_peer;

enum class PlaintextLayout : int { kDense, kCyclic, kContiguous, kStrided };
void check_compressed(const torch::Tensor& ciphertext,
                      const torch::Tensor& plaintext,
                      const char* operation) {
  TORCH_CHECK(ciphertext.dim() == 3 && plaintext.dim() == 3,
              operation,
              " requires rank-three views");
  TORCH_CHECK(ciphertext.size(1) == plaintext.size(1),
              operation,
              " limb counts differ");
  TORCH_CHECK(plaintext.size(0) == ciphertext.size(0) || plaintext.size(0) == 1,
              operation,
              " batch counts differ");
  TORCH_CHECK(plaintext.size(2) > 0 && plaintext.size(2) < ciphertext.size(2) &&
                  (plaintext.size(2) & (plaintext.size(2) - 1)) == 0 &&
                  ciphertext.size(2) % plaintext.size(2) == 0,
              operation,
              " compressed support must be a power-of-two divisor of N");
}

template <typename scalar_t, PlaintextLayout layout>
void plaintext_loop(torch::Tensor out,
                    const torch::Tensor ciphertext,
                    const torch::Tensor plaintext,
                    const torch::Tensor implicit,
                    const torch::Tensor params) {
  scalar_t* output = out.data_ptr<scalar_t>();
  const scalar_t* encrypted = ciphertext.data_ptr<scalar_t>();
  const scalar_t* prepared = plaintext.data_ptr<scalar_t>();
  const scalar_t* parameter_rows = params.data_ptr<scalar_t>();
  const scalar_t* implicit_values = nullptr;
  if constexpr (layout == PlaintextLayout::kStrided) {
    implicit_values = implicit.data_ptr<scalar_t>();
  }
  const int64_t batch_count = ciphertext.size(0);
  const int64_t limb_count = ciphertext.size(1);
  const int64_t coefficient_count = ciphertext.size(2);
  const int64_t support_count = plaintext.size(2);
  const int64_t repeat = coefficient_count / support_count;
  const int64_t plaintext_batch_count = plaintext.size(0);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t ct_stride0 = ciphertext.stride(0);
  const int64_t ct_stride1 = ciphertext.stride(1);
  const int64_t ct_stride2 = ciphertext.stride(2);
  const int64_t pt_stride0 = plaintext.stride(0);
  const int64_t pt_stride1 = plaintext.stride(1);
  const int64_t pt_stride2 = plaintext.stride(2);
  const int64_t out_stride0 = out.stride(0);
  const int64_t out_stride1 = out.stride(1);
  const int64_t out_stride2 = out.stride(2);
  const int64_t elements = batch_count * limb_count * coefficient_count;
  at::parallel_for(
      0,
      elements,
      fhelium::cpu::adaptive_grain(elements),
      [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        while (index < end) {
          const int64_t batch_limb = index / coefficient_count;
          const int64_t limb = batch_limb % limb_count;
          const int64_t batch = batch_limb / limb_count;
          const int64_t coefficient_begin =
              index - batch_limb * coefficient_count;
          const int64_t coefficient_end = std::min<int64_t>(
              coefficient_count, end - batch_limb * coefficient_count);
          const int64_t plaintext_batch =
              plaintext_batch_count == batch_count ? batch : 0;
          const auto constants =
              fhelium::cpu::load_constants(parameter_rows,
                                           parameter_row_stride,
                                           parameter_limb_stride,
                                           limb);
          const scalar_t* encrypted_row =
              encrypted + batch * ct_stride0 + limb * ct_stride1;
          const scalar_t* prepared_row =
              prepared + plaintext_batch * pt_stride0 + limb * pt_stride1;
          scalar_t* output_row =
              output + batch * out_stride0 + limb * out_stride1;
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            scalar_t plaintext_value;
            if constexpr (layout == PlaintextLayout::kDense) {
              plaintext_value = prepared_row[coefficient * pt_stride2];
            } else if constexpr (layout == PlaintextLayout::kCyclic) {
              plaintext_value =
                  prepared_row[(coefficient % support_count) * pt_stride2];
            } else if constexpr (layout == PlaintextLayout::kContiguous) {
              plaintext_value =
                  prepared_row[(coefficient / repeat) * pt_stride2];
            } else {
              plaintext_value =
                  implicit_values[plaintext_batch * limb_count + limb];
              if (coefficient % repeat == 0) {
                plaintext_value =
                    prepared_row[(coefficient / repeat) * pt_stride2];
              }
            }
            scalar_t value =
                fhelium::cpu::multiply(encrypted_row[coefficient * ct_stride2],
                                       constants.r2,
                                       constants);
            value = fhelium::cpu::add_lazy(
                value, plaintext_value, constants.twice_modulus);
            value = fhelium::cpu::reduce(value, constants);
            output_row[coefficient * out_stride2] =
                fhelium::cpu::reduce_to_standard(value,
                                                 constants.twice_modulus);
          }
          index = batch_limb * coefficient_count + coefficient_end;
        }
      });
}

template <PlaintextLayout layout>
void add_plaintext_cpu_(torch::Tensor ciphertext_component,
                        const torch::Tensor prepared_plaintext,
                        const torch::Tensor implicit_plaintext,
                        const torch::Tensor rns_params,
                        const char* operation) {
  check_cpu_peer(
      ciphertext_component, prepared_plaintext, operation, "plaintext");
  check_cpu_peer(ciphertext_component, rns_params, operation, "rns_params");
  auto ciphertext =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext =
      view_rns_batch_3d(prepared_plaintext, "prepared_plaintext");
  if constexpr (layout == PlaintextLayout::kDense) {
    check_rns_binary_3d(ciphertext, plaintext, operation, true);
  } else {
    check_compressed(ciphertext, plaintext, operation);
  }
  check_rns_parameter_rows(ciphertext, rns_params, operation);
  if constexpr (layout == PlaintextLayout::kStrided) {
    check_cpu_peer(ciphertext_component,
                   implicit_plaintext,
                   operation,
                   "implicit_plaintext");
    TORCH_CHECK(
        implicit_plaintext.numel() == plaintext.size(0) * plaintext.size(1) &&
            (implicit_plaintext.dim() == 1 || implicit_plaintext.dim() == 2) &&
            implicit_plaintext.is_contiguous(),
        operation,
        " implicit plaintext shape mismatch");
  }
  at::assert_no_internal_overlap(ciphertext);
  at::assert_no_overlap(ciphertext, plaintext);
  at::assert_no_overlap(ciphertext, rns_params);
  if constexpr (layout == PlaintextLayout::kStrided) {
    at::assert_no_overlap(ciphertext, implicit_plaintext);
  }
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(), "ckks_plaintext_cpu", [&] {
        plaintext_loop<scalar_t, layout>(
            ciphertext, ciphertext, plaintext, implicit_plaintext, rns_params);
      });
}

template <PlaintextLayout layout>
torch::Tensor add_plaintext_cpu(const torch::Tensor ciphertext_component,
                                const torch::Tensor prepared_plaintext,
                                const torch::Tensor implicit_plaintext,
                                const torch::Tensor rns_params,
                                const char* operation) {
  auto out = ciphertext_component.clone();
  add_plaintext_cpu_<layout>(
      out, prepared_plaintext, implicit_plaintext, rns_params, operation);
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CPU, m) {
  const auto empty = torch::Tensor{};
  m.impl(
      "add_prepared_plaintext_component",
      [empty](
          const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return add_plaintext_cpu<PlaintextLayout::kDense>(
            a, b, empty, p, "ckks_add_prepared_plaintext_component");
      });
  m.impl(
      "add_prepared_plaintext_component_",
      [empty](torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        add_plaintext_cpu_<PlaintextLayout::kDense>(
            a, b, empty, p, "ckks_add_prepared_plaintext_component");
      });
  m.impl(
      "add_cyclic_compressed_plaintext_component",
      [empty](
          const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return add_plaintext_cpu<PlaintextLayout::kCyclic>(
            a, b, empty, p, "ckks_add_cyclic_compressed_plaintext_component");
      });
  m.impl(
      "add_cyclic_compressed_plaintext_component_",
      [empty](torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        add_plaintext_cpu_<PlaintextLayout::kCyclic>(
            a, b, empty, p, "ckks_add_cyclic_compressed_plaintext_component");
      });
  m.impl(
      "add_contiguous_compressed_plaintext_component",
      [empty](
          const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return add_plaintext_cpu<PlaintextLayout::kContiguous>(
            a,
            b,
            empty,
            p,
            "ckks_add_contiguous_compressed_plaintext_component");
      });
  m.impl(
      "add_contiguous_compressed_plaintext_component_",
      [empty](torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        add_plaintext_cpu_<PlaintextLayout::kContiguous>(
            a,
            b,
            empty,
            p,
            "ckks_add_contiguous_compressed_plaintext_component");
      });
  m.impl("add_strided_plaintext_component",
         [](const torch::Tensor a,
            const torch::Tensor b,
            const torch::Tensor implicit,
            const torch::Tensor p) {
           return add_plaintext_cpu<PlaintextLayout::kStrided>(
               a, b, implicit, p, "ckks_add_strided_plaintext_component");
         });
  m.impl("add_strided_plaintext_component_",
         [](torch::Tensor a,
            const torch::Tensor b,
            const torch::Tensor implicit,
            const torch::Tensor p) {
           add_plaintext_cpu_<PlaintextLayout::kStrided>(
               a, b, implicit, p, "ckks_add_strided_plaintext_component");
         });
}
