#pragma once

#include <c10/util/Exception.h>

#include <cstdint>

namespace fhelium::detail {

// Physical key rows read by one QP operand. Operand rows `[0, q_rows)` read
// key rows `key_row_start + r`; the trailing auxiliary rows read
// `p_row_start + (r - q_rows)`. A negative `key_p_row_start` selects one
// contiguous key interval for every operand row.
struct KeyRowMap {
  int64_t q_rows;
  int64_t p_row_start;
};

inline KeyRowMap key_row_map(int64_t operand_rows,
                             int64_t key_rows,
                             int64_t key_row_start,
                             int64_t key_p_row_start,
                             const char* operation) {
  if (key_p_row_start < 0) {
    TORCH_CHECK(key_row_start >= 0 && key_row_start + operand_rows <= key_rows,
                operation,
                " active rows exceed the key row extent");
    return {operand_rows, key_row_start + operand_rows};
  }
  const int64_t p_rows = key_rows - key_p_row_start;
  const int64_t q_rows = operand_rows - p_rows;
  TORCH_CHECK(p_rows > 0 && q_rows >= 0 && key_row_start >= 0 &&
                  key_row_start + q_rows <= key_p_row_start,
              operation,
              " active Q rows or auxiliary rows exceed the key row extent");
  return {q_rows, key_p_row_start};
}

}  // namespace fhelium::detail
