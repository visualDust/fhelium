#pragma once

// libfhelium_torch registers the fhelium Torch operator schemas and their
// kernels with c10::Dispatcher from static initializers that run when the
// library is loaded. C++ programs link FHElium::torch and call
// load_torch_operators() so that the linker keeps the library as a dependency;
// Python loads the same file with torch.ops.load_library.

#if defined(_WIN32)
#if defined(FHELIUM_BUILDING_TORCH_LIBRARY)
#define FHELIUM_TORCH_API __declspec(dllexport)
#else
#define FHELIUM_TORCH_API __declspec(dllimport)
#endif
#else
#define FHELIUM_TORCH_API __attribute__((visibility("default")))
#endif

namespace fhelium {

// Returns after the library's operator registrations have completed.
FHELIUM_TORCH_API void load_torch_operators();

}  // namespace fhelium
