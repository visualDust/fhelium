#include "fhelium/torch_library.h"

namespace fhelium {

// Static registrations complete while the library loads, before any caller can
// reach this function.
void load_torch_operators() {}

}  // namespace fhelium
