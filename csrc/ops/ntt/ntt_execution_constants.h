#pragma once

namespace fhelium::ntt {

// Shared-memory storage is a compiled kernel resource, so the native layer is
// the single source of truth for its maximum and production default. The
// current 256-coefficient tile covers at most eight radix-2-equivalent
// transform bits. Diagnostic operators may select a smaller logical region,
// but cannot exceed the compiled tile.
inline constexpr int kNttMaxSharedMemoryLogN = 8;
inline constexpr int kNttDefaultSharedMemoryLogN = kNttMaxSharedMemoryLogN;
inline constexpr int kNttSharedMemoryTileSize = 1 << kNttMaxSharedMemoryLogN;

// A strict fixed-radix schedule can contribute at most four whole digits to
// the eight-bit budget: four radix-4 digits, two radix-8 digits, or two
// radix-16 digits.
inline constexpr int kPowerOfTwoRadixMaxSharedDigits = 4;

}  // namespace fhelium::ntt
