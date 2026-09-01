#pragma once

#include <torch/torch.h>

namespace fhelium::ckks::cpu {

inline void check_cpu_peer(const torch::Tensor& reference,
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

}  // namespace fhelium::ckks::cpu
