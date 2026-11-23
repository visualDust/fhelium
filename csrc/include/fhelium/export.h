#pragma once

// FHELIUM_API marks functions exported from the libfhelium shared library.
// The library build defines FHELIUM_BUILDING_LIBRARY; consumers import the
// same declarations.
#if defined(_WIN32)
#if defined(FHELIUM_BUILDING_LIBRARY)
#define FHELIUM_API __declspec(dllexport)
#else
#define FHELIUM_API __declspec(dllimport)
#endif
#else
#define FHELIUM_API __attribute__((visibility("default")))
#endif
