#pragma once

// Genuine strict fixed-radix NTT entry points. The validated root-table width
// selects the radix; the digit count follows uniquely from the ring size.
// Shared-memory coverage is native-owned and is overridable only through the
// separate diagnostic operator namespace.

template <typename scalar_t>
void launch_forward_ntt_montgomery_power_of_two_radix_compact_hip(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int radix,
    const int shared_memory_log_n) {
  const auto device_id = a.device().index();
  hipSetDevice(device_id);
  auto stream = c10::hip::getCurrentHIPStream(device_id);
  launch_forward_ntt_power_of_two_radix_hip<scalar_t>(a,
                                                       outer_twiddles,
                                                       radix_root_powers,
                                                       rns_params,
                                                       radix,
                                                       shared_memory_log_n,
                                                       stream);
}

void forward_ntt_montgomery_power_of_two_radix_compact_inplace_hip(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int64_t radix,
    const int64_t shared_memory_log_n) {
  auto batch_rows = view_rns_batch_3d(a, "montgomery_residues");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(),
      "forward_ntt_montgomery_power_of_two_radix_compact_hip",
      ([&] {
        launch_forward_ntt_montgomery_power_of_two_radix_compact_hip<scalar_t>(
            batch_rows,
            outer_twiddles,
            radix_root_powers,
            rns_params,
            static_cast<int>(radix),
            static_cast<int>(shared_memory_log_n));
      }));
}

template <typename scalar_t>
void launch_forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_hip(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int radix,
    const int shared_memory_log_n) {
  const auto params_acc = FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2);
  const auto device_id = a.device().index();
  hipSetDevice(device_id);
  auto stream = c10::hip::getCurrentHIPStream(device_id);

  const int rows = static_cast<int>(a.size(1));
  const int N = static_cast<int>(a.size(2));
  const dim3 grid(rows, N / kHipBlockSize, a.size(0));
  auto a_acc = FHELIUM_HIP_ACCESSOR32(a, scalar_t, 3);
  ntt_to_montgomery_inplace_kernel<scalar_t>
      <<<grid, kHipBlockSize, 0, stream>>>(a_acc, params_acc);

  launch_forward_ntt_power_of_two_radix_hip<scalar_t>(a,
                                                       outer_twiddles,
                                                       radix_root_powers,
                                                       rns_params,
                                                       radix,
                                                       shared_memory_log_n,
                                                       stream);
}

template <typename scalar_t>
void launch_forward_ntt_to_montgomery_power_of_two_radix_compact_out_hip(
    const torch::Tensor a,
    torch::Tensor out,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int radix,
    const int shared_memory_log_n) {
  const auto params_acc = FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2);
  const auto device_id = a.device().index();
  hipSetDevice(device_id);
  auto stream = c10::hip::getCurrentHIPStream(device_id);

  const int rows = static_cast<int>(a.size(1));
  const int N = static_cast<int>(a.size(2));
  const dim3 grid(rows, N / kHipBlockSize, a.size(0));
  const auto a_acc = FHELIUM_HIP_ACCESSOR32(a, scalar_t, 3);
  auto out_acc = FHELIUM_HIP_ACCESSOR32(out, scalar_t, 3);
  ntt_to_montgomery_out_kernel<scalar_t>
      <<<grid, kHipBlockSize, 0, stream>>>(out_acc, a_acc, params_acc);

  launch_forward_ntt_power_of_two_radix_hip<scalar_t>(out,
                                                       outer_twiddles,
                                                       radix_root_powers,
                                                       rns_params,
                                                       radix,
                                                       shared_memory_log_n,
                                                       stream);
}

void forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_hip(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int64_t radix,
    const int64_t shared_memory_log_n) {
  auto batch_rows = view_rns_batch_3d(a, "standard_residues");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(),
      "forward_ntt_to_montgomery_power_of_two_radix_compact_hip",
      ([&] {
        launch_forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_hip<
            scalar_t>(batch_rows,
                      outer_twiddles,
                      radix_root_powers,
                      rns_params,
                      static_cast<int>(radix),
                      static_cast<int>(shared_memory_log_n));
      }));
}

torch::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact_hip(
    const torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int64_t radix,
    const int64_t shared_memory_log_n) {
  torch::Tensor out = torch::empty_like(a);
  const auto batch_rows = view_rns_batch_3d(a, "standard_residues");
  auto out_batch_rows = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(),
      "forward_ntt_to_montgomery_power_of_two_radix_compact_hip",
      ([&] {
        launch_forward_ntt_to_montgomery_power_of_two_radix_compact_out_hip<
            scalar_t>(batch_rows,
                      out_batch_rows,
                      outer_twiddles,
                      radix_root_powers,
                      rns_params,
                      static_cast<int>(radix),
                      static_cast<int>(shared_memory_log_n));
      }));
  return out;
}
