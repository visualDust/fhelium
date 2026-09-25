#include <torch/library.h>

// Unnormalized compact radix-2 stages on Montgomery residues. The half-open
// interval indexes forward DIF or inverse DIT stages; kernel launches between
// stage groups provide the required global synchronization. Representation
// conversion and inverse normalization belong to the surrounding schedule.
TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "compact_ntt_stage_range_(Tensor(a!) residues, Tensor compact_twiddles, "
      "Tensor rns_params, bool inverse, int start_stage, int end_stage, "
      "int grouped_stage_count) -> ()");
}
