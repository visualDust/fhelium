#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"

namespace {

enum class PlaintextLayout : int { kDense, kCyclic, kContiguous, kStrided };

void check_cpu_peer(const torch::Tensor& reference,
                    const torch::Tensor& peer,
                    const char* operation,
                    const char* peer_name) {
  TORCH_CHECK(reference.device().is_cpu(), operation, " requires CPU tensors");
  TORCH_CHECK(peer.device() == reference.device(),
              operation,
              " requires ",
              peer_name,
              " on the operand CPU device");
  TORCH_CHECK(peer.scalar_type() == reference.scalar_type(),
              operation,
              " requires ",
              peer_name,
              " with the operand dtype");
}

void check_compressed(const torch::Tensor& ciphertext,
                      const torch::Tensor& plaintext,
                      const char* operation) {
  TORCH_CHECK(ciphertext.dim() == 3 && plaintext.dim() == 3,
              operation,
              " requires canonical rank-three views");
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
                fhelium::cpu::canonicalize(value, constants.twice_modulus);
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

template <typename scalar_t, bool nearest>
void rescale_loop(torch::Tensor out,
                  const torch::Tensor remaining,
                  const torch::Tensor inverse,
                  const torch::Tensor dropped,
                  const torch::Tensor params,
                  int64_t half_drop_prime) {
  scalar_t* output = out.data_ptr<scalar_t>();
  const scalar_t* input = remaining.data_ptr<scalar_t>();
  const scalar_t* inverse_values = inverse.data_ptr<scalar_t>();
  const scalar_t* dropped_values = dropped.data_ptr<scalar_t>();
  const scalar_t* parameter_rows = params.data_ptr<scalar_t>();
  const int64_t batch_count = remaining.size(0);
  const int64_t limbs = remaining.size(1);
  const int64_t coefficients = remaining.size(2);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t inverse_stride = inverse.stride(0);
  const int64_t input_stride0 = remaining.stride(0);
  const int64_t input_stride1 = remaining.stride(1);
  const int64_t input_stride2 = remaining.stride(2);
  const int64_t dropped_stride0 = dropped.stride(0);
  const int64_t dropped_stride1 = dropped.stride(1);
  const int64_t out_stride0 = out.stride(0);
  const int64_t out_stride1 = out.stride(1);
  const int64_t out_stride2 = out.stride(2);
  const int64_t elements = batch_count * limbs * coefficients;
  at::parallel_for(
      0,
      elements,
      fhelium::cpu::adaptive_grain(elements),
      [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        while (index < end) {
          const int64_t batch_limb = index / coefficients;
          const int64_t limb = batch_limb % limbs;
          const int64_t batch = batch_limb / limbs;
          const int64_t coefficient_begin = index - batch_limb * coefficients;
          const int64_t coefficient_end =
              std::min<int64_t>(coefficients, end - batch_limb * coefficients);
          const scalar_t* dropped_row =
              dropped_values + batch * dropped_stride0;
          const auto constants =
              fhelium::cpu::load_constants(parameter_rows,
                                           parameter_row_stride,
                                           parameter_limb_stride,
                                           limb);
          const scalar_t* input_row =
              input + batch * input_stride0 + limb * input_stride1;
          scalar_t* output_row =
              output + batch * out_stride0 + limb * out_stride1;
          const scalar_t inverse_value = inverse_values[limb * inverse_stride];
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            const scalar_t dropped_value =
                dropped_row[coefficient * dropped_stride1];
            scalar_t quotient = static_cast<scalar_t>(
                input_row[coefficient * input_stride2] - dropped_value);
            quotient = fhelium::cpu::multiply_split(
                quotient, inverse_value, constants);
            if constexpr (nearest) {
              quotient = static_cast<scalar_t>(
                  quotient + (dropped_value > half_drop_prime ? 1 : 0));
            }
            output_row[coefficient * out_stride2] =
                fhelium::cpu::canonicalize(quotient, constants.twice_modulus);
          }
          index = batch_limb * coefficients + coefficient_end;
        }
      });
}

template <bool nearest>
void rescale_cpu_(torch::Tensor remaining_residues,
                  const torch::Tensor inverse,
                  const torch::Tensor dropped_residue,
                  const torch::Tensor rns_params,
                  int64_t half_drop_prime,
                  const char* operation) {
  for (const auto& peer : {inverse, dropped_residue, rns_params}) {
    check_cpu_peer(remaining_residues, peer, operation, "peer tensor");
  }
  auto remaining = view_rns_batch_3d(remaining_residues, "remaining_residues");
  const auto dropped =
      view_coefficient_batch_2d(dropped_residue, "dropped_residue");
  check_rns_parameter_rows(remaining, rns_params, operation);
  check_rns_row_vector(inverse, remaining.size(1), operation, "inverse");
  TORCH_CHECK(dropped.size(0) == remaining.size(0) &&
                  dropped.size(1) == remaining.size(2),
              operation,
              " dropped residue shape mismatch");
  at::assert_no_internal_overlap(remaining);
  at::assert_no_overlap(remaining, inverse);
  at::assert_no_overlap(remaining, dropped);
  at::assert_no_overlap(remaining, rns_params);
  AT_DISPATCH_INTEGRAL_TYPES(
      remaining_residues.scalar_type(), "ckks_rescale_cpu", [&] {
        rescale_loop<scalar_t, nearest>(remaining,
                                        remaining,
                                        inverse,
                                        dropped,
                                        rns_params,
                                        half_drop_prime);
      });
}

template <bool nearest>
torch::Tensor rescale_cpu(const torch::Tensor remaining,
                          const torch::Tensor inverse,
                          const torch::Tensor dropped,
                          const torch::Tensor params,
                          int64_t half_drop_prime,
                          const char* operation) {
  auto out = remaining.clone();
  rescale_cpu_<nearest>(
      out, inverse, dropped, params, half_drop_prime, operation);
  return out;
}

torch::Tensor coefficient_galois_cpu(const torch::Tensor residues,
                                     const torch::Tensor source_indices,
                                     const torch::Tensor source_sign,
                                     const torch::Tensor twice_modulus) {
  constexpr const char* operation = "apply_coefficient_galois_automorphism";
  for (const auto& peer : {source_indices, source_sign, twice_modulus}) {
    TORCH_CHECK(peer.device() == residues.device(),
                operation,
                " requires tables on the operand CPU device");
  }
  TORCH_CHECK(residues.device().is_cpu(), operation, " requires CPU tensors");
  const auto input = view_rns_batch_3d(residues, "residues");
  TORCH_CHECK(source_indices.scalar_type() == torch::kInt32 &&
                  source_sign.scalar_type() == torch::kInt8 &&
                  source_indices.dim() == 1 && source_sign.dim() == 1 &&
                  source_indices.size(0) == input.size(2) &&
                  source_sign.size(0) == input.size(2),
              operation,
              " gather table shape or dtype mismatch");
  check_rns_row_vector(
      twice_modulus, input.size(1), operation, "twice_modulus");
  check_cpu_peer(residues, twice_modulus, operation, "twice_modulus");
  auto out = torch::empty_like(residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), operation, [&] {
    const scalar_t* source = input.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const int32_t* indices = source_indices.data_ptr<int32_t>();
    const int8_t* signs = source_sign.data_ptr<int8_t>();
    const scalar_t* twice = twice_modulus.data_ptr<scalar_t>();
    const int64_t index_stride = source_indices.stride(0);
    const int64_t sign_stride = source_sign.stride(0);
    const int64_t twice_stride = twice_modulus.stride(0);
    const int64_t batch_count = input.size(0);
    const int64_t limbs = input.size(1);
    const int64_t coefficients = input.size(2);
    const int64_t source_stride0 = input.stride(0);
    const int64_t source_stride1 = input.stride(1);
    const int64_t source_stride2 = input.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    const int64_t elements = batch_count * limbs * coefficients;
    for (int64_t coefficient = 0; coefficient < coefficients; ++coefficient) {
      const int64_t source_index = indices[coefficient * index_stride];
      TORCH_CHECK(source_index >= 0 && source_index < coefficients,
                  operation,
                  " source index is out of bounds");
    }
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_limb = index / coefficients;
            const int64_t limb = batch_limb % limbs;
            const int64_t batch = batch_limb / limbs;
            const int64_t coefficient_begin = index - batch_limb * coefficients;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficients, end - batch_limb * coefficients);
            const scalar_t twice_modulus_value = twice[limb * twice_stride];
            const scalar_t* source_rows = source + batch * source_stride0;
            scalar_t* result_row =
                result + batch * result_stride0 + limb * result_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              const int64_t source_index = indices[coefficient * index_stride];
              scalar_t value = source_rows[limb * source_stride1 +
                                           source_index * source_stride2];
              if (signs[coefficient * sign_stride] == -1) value = -value;
              value = fhelium::cpu::shift_positive(value, twice_modulus_value);
              result_row[coefficient * result_stride2] =
                  fhelium::cpu::canonicalize(value, twice_modulus_value);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

torch::Tensor ntt_galois_cpu(const torch::Tensor residues,
                             const torch::Tensor source_indices) {
  constexpr const char* operation = "apply_ntt_galois_automorphism";
  TORCH_CHECK(residues.device().is_cpu() &&
                  source_indices.device() == residues.device() &&
                  source_indices.scalar_type() == torch::kInt32,
              operation,
              " requires CPU residues and int32 CPU indices");
  const auto input = view_rns_batch_3d(residues, "residues_ntt");
  TORCH_CHECK(
      source_indices.dim() == 1 && source_indices.size(0) == input.size(2),
      operation,
      " source index count mismatch");
  auto out = torch::empty_like(residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), operation, [&] {
    const scalar_t* source = input.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const int32_t* indices = source_indices.data_ptr<int32_t>();
    const int64_t index_stride = source_indices.stride(0);
    const int64_t batch_count = input.size(0);
    const int64_t limbs = input.size(1);
    const int64_t coefficients = input.size(2);
    const int64_t source_stride0 = input.stride(0);
    const int64_t source_stride1 = input.stride(1);
    const int64_t source_stride2 = input.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    const int64_t elements = batch_count * limbs * coefficients;
    for (int64_t coefficient = 0; coefficient < coefficients; ++coefficient) {
      const int64_t source_index = indices[coefficient * index_stride];
      TORCH_CHECK(source_index >= 0 && source_index < coefficients,
                  operation,
                  " source index is out of bounds");
    }
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_limb = index / coefficients;
            const int64_t limb = batch_limb % limbs;
            const int64_t batch = batch_limb / limbs;
            const int64_t coefficient_begin = index - batch_limb * coefficients;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficients, end - batch_limb * coefficients);
            const scalar_t* source_rows = source + batch * source_stride0;
            scalar_t* result_row =
                result + batch * result_stride0 + limb * result_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              result_row[coefficient * result_stride2] =
                  source_rows[limb * source_stride1 +
                              indices[coefficient * index_stride] *
                                  source_stride2];
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

torch::Tensor keyswitch_moddown_cpu(const torch::Tensor q_residues,
                                    const torch::Tensor p_residues,
                                    const torch::Tensor inverse,
                                    const torch::Tensor rns_params) {
  constexpr const char* operation = "keyswitch_moddown_qp_to_q";
  for (const auto& peer : {p_residues, inverse, rns_params}) {
    check_cpu_peer(q_residues, peer, operation, "peer tensor");
  }
  const auto q = view_rns_batch_3d(q_residues, "q_residues");
  const auto p = view_rns_batch_3d(p_residues, "p_residues");
  TORCH_CHECK(q.size(0) == p.size(0) && q.size(2) == p.size(2),
              operation,
              " Q/P batch and coefficient shapes differ");
  TORCH_CHECK(
      rns_params.dim() == 2 && rns_params.size(1) == q.size(1) + p.size(1),
      operation,
      " RNS parameter row count mismatch");
  TORCH_CHECK(inverse.dim() == 2 && inverse.size(0) == p.size(1) &&
                  inverse.size(1) >= q.size(1) + p.size(1) - 1,
              operation,
              " inverse table shape mismatch");
  auto out = torch::empty_like(q_residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(q_residues.scalar_type(), operation, [&] {
    const scalar_t* q_values = q.data_ptr<scalar_t>();
    const scalar_t* p_values = p.data_ptr<scalar_t>();
    const scalar_t* inverse_values = inverse.data_ptr<scalar_t>();
    const scalar_t* parameter_rows = rns_params.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const int64_t batch_count = q.size(0);
    const int64_t q_count = q.size(1);
    const int64_t p_count = p.size(1);
    const int64_t coefficients = q.size(2);
    const int64_t parameter_row_stride = rns_params.stride(0);
    const int64_t parameter_limb_stride = rns_params.stride(1);
    const int64_t inverse_drop_stride = inverse.stride(0);
    const int64_t inverse_limb_stride = inverse.stride(1);
    const int64_t q_stride0 = q.stride(0);
    const int64_t q_stride1 = q.stride(1);
    const int64_t q_stride2 = q.stride(2);
    const int64_t p_stride0 = p.stride(0);
    const int64_t p_stride1 = p.stride(1);
    const int64_t p_stride2 = p.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    const int64_t elements = batch_count * coefficients;
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          c10::SmallVector<scalar_t, 8> p_chain(static_cast<size_t>(p_count));
          int64_t index = begin;
          while (index < end) {
            const int64_t batch = index / coefficients;
            const int64_t coefficient_begin = index - batch * coefficients;
            const int64_t coefficient_end =
                std::min<int64_t>(coefficients, end - batch * coefficients);
            const scalar_t* q_rows = q_values + batch * q_stride0;
            const scalar_t* p_rows = p_values + batch * p_stride0;
            scalar_t* result_rows = result + batch * result_stride0;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              for (int64_t row = 0; row < p_count; ++row) {
                p_chain[static_cast<size_t>(row)] =
                    p_rows[row * p_stride1 + coefficient * p_stride2];
              }
              for (int64_t row = p_count - 2; row >= 0; --row) {
                scalar_t value = p_chain[static_cast<size_t>(row)];
                const int64_t parameter_row = q_count + row;
                const auto constants =
                    fhelium::cpu::load_constants(parameter_rows,
                                                 parameter_row_stride,
                                                 parameter_limb_stride,
                                                 parameter_row);
                for (int64_t lower = p_count - 1; lower > row; --lower) {
                  const scalar_t difference = fhelium::cpu::subtract_lazy(
                      value,
                      p_chain[static_cast<size_t>(lower)],
                      constants.twice_modulus);
                  value = fhelium::cpu::multiply_split(
                      difference,
                      inverse_values[(p_count - lower - 1) *
                                         inverse_drop_stride +
                                     (row + q_count) * inverse_limb_stride],
                      constants);
                }
                p_chain[static_cast<size_t>(row)] = value;
              }

              for (int64_t row = 0; row < q_count; ++row) {
                const auto constants =
                    fhelium::cpu::load_constants(parameter_rows,
                                                 parameter_row_stride,
                                                 parameter_limb_stride,
                                                 row);
                scalar_t value = fhelium::cpu::multiply(
                    q_rows[row * q_stride1 + coefficient * q_stride2],
                    constants.r2,
                    constants);
                for (int64_t p_row = p_count - 1; p_row >= 0; --p_row) {
                  const scalar_t p_value_montgomery = fhelium::cpu::multiply(
                      p_chain[static_cast<size_t>(p_row)],
                      constants.r2,
                      constants);
                  value = fhelium::cpu::subtract_lazy(
                      value, p_value_montgomery, constants.twice_modulus);
                  value = fhelium::cpu::multiply_split(
                      value,
                      inverse_values[(p_count - p_row - 1) *
                                         inverse_drop_stride +
                                     row * inverse_limb_stride],
                      constants);
                }
                value = fhelium::cpu::reduce(value, constants);
                result_rows[row * result_stride1 +
                            coefficient * result_stride2] =
                    fhelium::cpu::canonicalize(value, constants.twice_modulus);
              }
            }
            index = batch * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

void keyswitch_accumulate_cpu_(torch::Tensor accumulator0_qp,
                               torch::Tensor accumulator1_qp,
                               const torch::Tensor digit_qp,
                               const torch::Tensor key_digit,
                               const torch::Tensor rns_params,
                               const int64_t key_row_start) {
  constexpr const char* operation = "keyswitch_accumulate_digit_products";
  for (const auto& peer :
       {accumulator0_qp, accumulator1_qp, key_digit, rns_params}) {
    check_cpu_peer(digit_qp, peer, operation, "peer tensor");
  }
  auto accumulator0 = view_rns_batch_3d(accumulator0_qp, "accumulator0_qp");
  auto accumulator1 = view_rns_batch_3d(accumulator1_qp, "accumulator1_qp");
  const auto digit = view_rns_batch_3d(digit_qp, "digit_qp");
  check_rns_binary_3d(accumulator0, digit, operation, false);
  check_rns_binary_3d(accumulator1, digit, operation, false);
  check_rns_parameter_rows(digit, rns_params, operation);
  TORCH_CHECK(key_digit.dim() == 3 && key_digit.size(0) == 2 &&
                  key_row_start >= 0 &&
                  key_row_start + digit.size(1) <= key_digit.size(1) &&
                  key_digit.size(2) == digit.size(2),
              operation,
              " key digit shape or row interval mismatch");
  at::assert_no_internal_overlap(accumulator0);
  at::assert_no_internal_overlap(accumulator1);
  at::assert_no_overlap(accumulator0, accumulator1);
  for (const auto& read_only : {digit, key_digit, rns_params}) {
    at::assert_no_overlap(accumulator0, read_only);
    at::assert_no_overlap(accumulator1, read_only);
  }
  AT_DISPATCH_INTEGRAL_TYPES(digit_qp.scalar_type(), operation, [&] {
    scalar_t* out0 = accumulator0.data_ptr<scalar_t>();
    scalar_t* out1 = accumulator1.data_ptr<scalar_t>();
    const scalar_t* source = digit.data_ptr<scalar_t>();
    const scalar_t* key = key_digit.data_ptr<scalar_t>();
    const scalar_t* parameter_rows = rns_params.data_ptr<scalar_t>();
    const int64_t batch_count = digit.size(0);
    const int64_t limbs = digit.size(1);
    const int64_t coefficients = digit.size(2);
    const int64_t parameter_row_stride = rns_params.stride(0);
    const int64_t parameter_limb_stride = rns_params.stride(1);
    const int64_t digit_stride0 = digit.stride(0);
    const int64_t digit_stride1 = digit.stride(1);
    const int64_t digit_stride2 = digit.stride(2);
    const int64_t key_stride1 = key_digit.stride(1);
    const int64_t key_stride2 = key_digit.stride(2);
    const int64_t out0_stride0 = accumulator0.stride(0);
    const int64_t out0_stride1 = accumulator0.stride(1);
    const int64_t out0_stride2 = accumulator0.stride(2);
    const int64_t out1_stride0 = accumulator1.stride(0);
    const int64_t out1_stride1 = accumulator1.stride(1);
    const int64_t out1_stride2 = accumulator1.stride(2);
    const int64_t elements = batch_count * limbs * coefficients;
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_limb = index / coefficients;
            const int64_t limb = batch_limb % limbs;
            const int64_t batch = batch_limb / limbs;
            const int64_t coefficient_begin = index - batch_limb * coefficients;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficients, end - batch_limb * coefficients);
            const auto constants =
                fhelium::cpu::load_constants(parameter_rows,
                                             parameter_row_stride,
                                             parameter_limb_stride,
                                             limb);
            const scalar_t* source_row =
                source + batch * digit_stride0 + limb * digit_stride1;
            const int64_t key_row = key_row_start + limb;
            const int64_t key_component_stride = key_digit.stride(0);
            const scalar_t* key_row0 = key + key_row * key_stride1;
            const scalar_t* key_row1 =
                key + key_component_stride + key_row * key_stride1;
            scalar_t* out0_row =
                out0 + batch * out0_stride0 + limb * out0_stride1;
            scalar_t* out1_row =
                out1 + batch * out1_stride0 + limb * out1_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              const scalar_t value = source_row[coefficient * digit_stride2];
              const scalar_t product0 = fhelium::cpu::multiply(
                  value, key_row0[coefficient * key_stride2], constants);
              const scalar_t product1 = fhelium::cpu::multiply(
                  value, key_row1[coefficient * key_stride2], constants);
              out0_row[coefficient * out0_stride2] =
                  fhelium::cpu::add_lazy(out0_row[coefficient * out0_stride2],
                                         product0,
                                         constants.twice_modulus);
              out1_row[coefficient * out1_stride2] =
                  fhelium::cpu::add_lazy(out1_row[coefficient * out1_stride2],
                                         product1,
                                         constants.twice_modulus);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
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
  m.impl("rescale_drop_leading_prime_nearest",
         [](const torch::Tensor a,
            const torch::Tensor inv,
            const torch::Tensor dropped,
            const torch::Tensor p,
            int64_t half) {
           return rescale_cpu<true>(a,
                                    inv,
                                    dropped,
                                    p,
                                    half,
                                    "ckks_rescale_drop_leading_prime_nearest");
         });
  m.impl("rescale_drop_leading_prime_nearest_",
         [](torch::Tensor a,
            const torch::Tensor inv,
            const torch::Tensor dropped,
            const torch::Tensor p,
            int64_t half) {
           rescale_cpu_<true>(a,
                              inv,
                              dropped,
                              p,
                              half,
                              "ckks_rescale_drop_leading_prime_nearest");
         });
  m.impl(
      "rescale_drop_leading_prime_truncate",
      [](const torch::Tensor a,
         const torch::Tensor inv,
         const torch::Tensor dropped,
         const torch::Tensor p) {
        return rescale_cpu<false>(
            a, inv, dropped, p, 0, "ckks_rescale_drop_leading_prime_truncate");
      });
  m.impl(
      "rescale_drop_leading_prime_truncate_",
      [](torch::Tensor a,
         const torch::Tensor inv,
         const torch::Tensor dropped,
         const torch::Tensor p) {
        rescale_cpu_<false>(
            a, inv, dropped, p, 0, "ckks_rescale_drop_leading_prime_truncate");
      });
  m.impl("apply_coefficient_galois_automorphism", &coefficient_galois_cpu);
  m.impl("apply_ntt_galois_automorphism", &ntt_galois_cpu);
  m.impl("keyswitch_moddown_qp_to_q", &keyswitch_moddown_cpu);
  m.impl("keyswitch_accumulate_digit_products_", &keyswitch_accumulate_cpu_);
}
