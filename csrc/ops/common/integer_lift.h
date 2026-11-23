#pragma once

#include <c10/macros/Macros.h>

#include <type_traits>

namespace fhelium {

// Residue of a signed integer modulo q = twice_modulus / 2 in the lazy range
// [0, 2q): x + q for -q <= x < q, otherwise x mod q in [0, q).
template <typename residue_t, typename integer_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE residue_t
lift_integer_residue(const integer_t value, const residue_t twice_modulus) {
  using wide_t = std::conditional_t<(sizeof(integer_t) > sizeof(residue_t)),
                                    integer_t,
                                    residue_t>;
  const wide_t q = static_cast<wide_t>(twice_modulus >> 1);
  const wide_t x = static_cast<wide_t>(value);
  if (x >= -q && x < q) {
    return static_cast<residue_t>(x + q);
  }
  const wide_t remainder = x % q;
  return static_cast<residue_t>(remainder < 0 ? remainder + q : remainder);
}

}  // namespace fhelium
