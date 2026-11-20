#pragma once

// libfhelium public C++ interface. Functions take and return ATen Tensors and
// select their CPU or GPU implementation from the device of their Tensor
// arguments: a GPU argument selects the GPU implementation, and CPU-only
// arguments select the CPU implementation. Each function corresponds to the
// Torch operator of the same name in the fhelium_ckks_ops, fhelium_rns_ops,
// fhelium_ntt_ops, or fhelium_ntt_diagnostic_ops namespace.

#include <fhelium/ckks.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_diagnostic.h>
#include <fhelium/rns.h>
#include <fhelium/rns_parameters.h>
