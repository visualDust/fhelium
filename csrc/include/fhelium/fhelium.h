#pragma once

// libfhelium public C++ interface. Functions take and return ATen Tensors and
// select their CPU or GPU implementation from the device of their Tensor
// arguments: a GPU argument selects the GPU implementation, and CPU-only
// arguments select the CPU implementation. Table and parameter functions
// compute on the host and return CPU Tensors or host values. Whole RNS, NTT,
// and CKKS operations and key generation compose these functions and read
// their numerical operands from the table structs of
// fhelium/operation_tables.h. Each function corresponds to
// the Torch operator of the same name in the fhelium_ckks_ops,
// fhelium_rns_ops, fhelium_ntt_ops, fhelium_ntt_diagnostic_ops, or
// fhelium_random_ops namespace.

#include <fhelium/ckks.h>
#include <fhelium/ckks_keys.h>
#include <fhelium/ckks_operations.h>
#include <fhelium/ckks_parameters.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_diagnostic.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/ntt_tables.h>
#include <fhelium/operation_tables.h>
#include <fhelium/random.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>
#include <fhelium/rns_parameters.h>
#include <fhelium/rns_tables.h>
