#include <fhelium/ckks_parameters.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace fhelium::ckks {
namespace {

struct PrimeSequence {
  int64_t prime_bits;
  int64_t ring_dimension;
  int64_t first;
  int64_t count;
};

#include "prime_catalog_data.inc"

// Little-endian base-2^32 digits of the product of `moduli`.
std::vector<uint64_t> product_limbs(c10::ArrayRef<int64_t> moduli) {
  std::vector<uint64_t> limbs{1};
  for (const int64_t value : moduli) {
    TORCH_CHECK_VALUE(value >= 2, "every entry in moduli must be at least 2");
    const auto factor = static_cast<uint64_t>(value);
    const uint64_t lo = factor & 0xffffffffu;
    const uint64_t hi = factor >> 32;
    std::vector<uint64_t> next(limbs.size() + 2, 0);
    for (size_t i = 0; i < limbs.size(); ++i) {
      uint64_t carry = 0;
      for (size_t j = 0; j < 2; ++j) {
        const uint64_t digit = j == 0 ? lo : hi;
        uint64_t& slot = next[i + j];
        const uint64_t product = limbs[i] * digit + slot + carry;
        slot = product & 0xffffffffu;
        carry = product >> 32;
      }
      size_t k = i + 2;
      while (carry != 0) {
        const uint64_t sum = next[k] + carry;
        next[k] = sum & 0xffffffffu;
        carry = sum >> 32;
        ++k;
      }
    }
    while (next.size() > 1 && next.back() == 0) {
      next.pop_back();
    }
    limbs = std::move(next);
  }
  return limbs;
}

// Compares two little-endian base-2^32 numbers without leading zero digits.
bool less_than(const std::vector<uint64_t>& lhs,
               const std::vector<uint64_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return lhs.size() < rhs.size();
  }
  return std::lexicographical_compare(
      lhs.rbegin(), lhs.rend(), rhs.rbegin(), rhs.rend());
}

#include "ckks_preset_data.inc"

struct SecurityRow {
  bool ternary;
  int64_t target_bits;
  int64_t ring_dimension;
  int64_t maximum_bits;
};

constexpr SecurityRow kSecurityRows[] = {
    {true, 128, 1024, 26},      {true, 128, 2048, 53},
    {true, 128, 4096, 106},     {true, 128, 8192, 214},
    {true, 128, 16384, 430},    {true, 128, 32768, 868},
    {true, 128, 65536, 1747},   {true, 128, 131072, 3523},
    {true, 192, 2048, 36},      {true, 192, 4096, 73},
    {true, 192, 8192, 147},     {true, 192, 16384, 297},
    {true, 192, 32768, 597},    {true, 192, 65536, 1199},
    {true, 192, 131072, 2411},  {true, 256, 2048, 27},
    {true, 256, 4096, 56},      {true, 256, 8192, 114},
    {true, 256, 16384, 230},    {true, 256, 32768, 462},
    {true, 256, 65536, 929},    {true, 256, 131072, 1866},
    {false, 128, 1024, 28},     {false, 128, 2048, 55},
    {false, 128, 4096, 108},    {false, 128, 8192, 216},
    {false, 128, 16384, 432},   {false, 128, 32768, 870},
    {false, 128, 65536, 1749},  {false, 128, 131072, 3525},
    {false, 192, 2048, 38},     {false, 192, 4096, 75},
    {false, 192, 8192, 149},    {false, 192, 16384, 299},
    {false, 192, 32768, 599},   {false, 192, 65536, 1201},
    {false, 192, 131072, 2413}, {false, 256, 2048, 30},
    {false, 256, 4096, 58},     {false, 256, 8192, 116},
    {false, 256, 16384, 232},   {false, 256, 32768, 464},
    {false, 256, 65536, 931},   {false, 256, 131072, 1868},
};

constexpr double kTableErrorStddev = 3.19;

template <size_t Size>
std::vector<int64_t> lookup(const PrimeSequence (&table)[Size],
                            int64_t prime_bits,
                            int64_t ring_dimension,
                            const char* kind) {
  for (const auto& entry : table) {
    if (entry.prime_bits == prime_bits &&
        entry.ring_dimension == ring_dimension) {
      return {kPrimeValues + entry.first,
              kPrimeValues + entry.first + entry.count};
    }
  }
  TORCH_CHECK_VALUE(false,
                    "The prime catalog has no ",
                    kind,
                    " primes of ",
                    prime_bits,
                    " bits for ring dimension ",
                    ring_dimension);
  return {};
}

}  // namespace

std::vector<int64_t> scaling_primes(int64_t prime_bits,
                                    int64_t ring_dimension) {
  return lookup(kScalingPrimes, prime_bits, ring_dimension, "scaling");
}

std::vector<int64_t> special_primes(int64_t prime_bits,
                                    int64_t ring_dimension) {
  return lookup(kSpecialPrimes, prime_bits, ring_dimension, "special");
}

std::vector<std::string> preset_names() {
  std::vector<std::string> names;
  for (const auto& preset : kPresets) {
    names.emplace_back(preset.name);
  }
  return names;
}

CkksParameters preset_parameters(std::string_view name) {
  const PresetData* preset = nullptr;
  for (const auto& candidate : kPresets) {
    if (name == candidate.name) {
      preset = &candidate;
    }
  }
  TORCH_CHECK_VALUE(
      preset != nullptr, "Unknown CKKS preset ", std::string(name));
  CkksParameters parameters;
  parameters.log_n = preset->log_n;
  parameters.q_moduli.assign(kPresetPrimes + preset->q_first,
                             kPresetPrimes + preset->q_first + preset->q_count);
  int64_t row = preset->row_first;
  for (int64_t i = 0; i < preset->base_count; ++i) {
    const int64_t size = kPresetBasisSizes[preset->base_first + i];
    parameters.q_bases.emplace_back(kPresetBasisRows + row,
                                    kPresetBasisRows + row + size);
    parameters.scales.push_back(kPresetScales[preset->base_first + i]);
    row += size;
  }
  parameters.p_moduli.assign(kPresetPrimes + preset->p_first,
                             kPresetPrimes + preset->p_first + preset->p_count);
  return parameters;
}

void validate_moduli(int64_t log_n, c10::ArrayRef<int64_t> moduli) {
  TORCH_CHECK_VALUE(log_n >= 1 && log_n <= 30,
                    "logN must be a positive integer of at most 30");
  const std::set<int64_t> distinct(moduli.begin(), moduli.end());
  TORCH_CHECK_VALUE(distinct.size() == moduli.size(),
                    "Q and P must contain distinct primes");
  const int64_t order = int64_t{2} << log_n;
  for (const int64_t q : moduli) {
    TORCH_CHECK_VALUE(q > 2 && q % 2 != 0,
                      "Every Q/P modulus must be an odd integer above two");
  }
  for (const int64_t q : moduli) {
    TORCH_CHECK_VALUE(
        (q - 1) % order == 0,
        "Every Q/P modulus must support the configured negacyclic NTT");
  }
}

int64_t modulus_bits(c10::ArrayRef<int64_t> moduli) {
  TORCH_CHECK_VALUE(!moduli.empty(),
                    "moduli must contain at least one modulus");
  auto limbs = product_limbs(moduli);
  // Subtract one, then take the bit length.
  for (auto& limb : limbs) {
    if (limb != 0) {
      --limb;
      break;
    }
    limb = 0xffffffffu;
  }
  while (limbs.size() > 1 && limbs.back() == 0) {
    limbs.pop_back();
  }
  int64_t bits = static_cast<int64_t>(limbs.size() - 1) * 32;
  for (uint64_t top = limbs.back(); top != 0; top >>= 1) {
    ++bits;
  }
  return bits;
}

std::vector<int64_t> key_digit_sizes(c10::ArrayRef<int64_t> q_moduli,
                                     c10::ArrayRef<int64_t> p_moduli) {
  TORCH_CHECK_VALUE(!p_moduli.empty(),
                    "p_moduli must contain at least one modulus");
  const auto p_product = product_limbs(p_moduli);
  std::vector<int64_t> sizes;
  size_t start = 0;
  while (start < q_moduli.size()) {
    size_t stop = start + 1;
    while (stop < q_moduli.size() &&
           less_than(product_limbs(q_moduli.slice(start, stop - start + 1)),
                     p_product)) {
      ++stop;
    }
    sizes.push_back(static_cast<int64_t>(stop - start));
    start = stop;
  }
  return sizes;
}

namespace {

std::vector<int64_t> gather(c10::ArrayRef<int64_t> moduli,
                            c10::ArrayRef<int64_t> ids) {
  std::vector<int64_t> result;
  result.reserve(ids.size());
  for (const auto id : ids) {
    result.push_back(moduli[id]);
  }
  return result;
}

// Number of greedy runs of `ids` whose products are below `limit`, or zero
// when one modulus reaches the limit.
size_t greedy_digit_count(c10::ArrayRef<int64_t> moduli,
                          c10::ArrayRef<int64_t> ids,
                          const std::vector<uint64_t>& limit) {
  size_t count = 0;
  size_t start = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    const std::vector<int64_t> single{moduli[ids[i]]};
    if (!less_than(product_limbs(single), limit)) {
      return 0;
    }
    if (i > start && !less_than(product_limbs(gather(
                                    moduli, ids.slice(start, i - start + 1))),
                                limit)) {
      ++count;
      start = i;
    }
  }
  return ids.empty() ? 0 : count + 1;
}

// Splits `ids` into `count` runs that minimize the largest run product.
std::vector<std::vector<int64_t>> minimax_digits(c10::ArrayRef<int64_t> moduli,
                                                 c10::ArrayRef<int64_t> ids,
                                                 size_t count) {
  const size_t n = ids.size();
  const auto run = [&](size_t start, size_t stop) {
    return product_limbs(gather(moduli, ids.slice(start, stop - start)));
  };
  struct Entry {
    bool set = false;
    std::vector<uint64_t> largest;
    size_t start = 0;
  };
  // best[k][stop]: the smallest largest product of k runs covering ids[:stop]
  // and the start of the last run; ties keep the smaller start.
  std::vector<std::vector<Entry>> best(count + 1, std::vector<Entry>(n + 1));
  best[0][0] = {true, {1}, 0};
  for (size_t k = 1; k <= count; ++k) {
    for (size_t stop = k; stop <= n - count + k; ++stop) {
      auto& entry = best[k][stop];
      for (size_t start = k - 1; start < stop; ++start) {
        const auto& previous = best[k - 1][start];
        if (!previous.set) {
          continue;
        }
        auto product = run(start, stop);
        auto largest =
            less_than(previous.largest, product) ? product : previous.largest;
        if (!entry.set || less_than(largest, entry.largest)) {
          entry = {true, std::move(largest), start};
        }
      }
    }
  }
  std::vector<std::vector<int64_t>> digits(count);
  size_t stop = n;
  for (size_t k = count; k > 0; --k) {
    const auto start = best[k][stop].start;
    digits[k - 1].assign(ids.begin() + start, ids.begin() + stop);
    stop = start;
  }
  return digits;
}

}  // namespace

KeyLayout plan_key_layout(c10::ArrayRef<int64_t> moduli,
                          c10::ArrayRef<int64_t> q_prime_ids) {
  std::vector<int64_t> eligible;
  for (int64_t id = 0; id < static_cast<int64_t>(moduli.size()); ++id) {
    if (std::find(q_prime_ids.begin(), q_prime_ids.end(), id) ==
        q_prime_ids.end()) {
      eligible.push_back(id);
    }
  }
  std::stable_sort(eligible.begin(), eligible.end(), [&](int64_t a, int64_t b) {
    return moduli[a] > moduli[b];
  });
  const auto q = static_cast<int64_t>(q_prime_ids.size());
  bool found = false;
  std::tuple<int64_t, int64_t, int64_t> best_cost;
  size_t best_count = 0;
  size_t best_width = 0;
  for (size_t width = 1; width <= eligible.size(); ++width) {
    const auto limit = product_limbs(
        gather(moduli, c10::ArrayRef<int64_t>(eligible.data(), width)));
    const auto k =
        static_cast<int64_t>(greedy_digit_count(moduli, q_prime_ids, limit));
    if (k == 0) {
      continue;
    }
    const auto w = static_cast<int64_t>(width);
    const std::tuple<int64_t, int64_t, int64_t> cost{
        (k + 1) * (q + w), k * (q + w), w};
    if (!found || cost < best_cost) {
      found = true;
      best_cost = cost;
      best_count = static_cast<size_t>(k);
      best_width = width;
    }
  }
  TORCH_CHECK_VALUE(found,
                    "the configured primes outside the active rows cannot "
                    "bound a key digit");
  KeyLayout layout;
  layout.q_prime_ids = q_prime_ids.vec();
  layout.p_prime_ids.assign(eligible.begin(), eligible.begin() + best_width);
  std::sort(layout.p_prime_ids.begin(), layout.p_prime_ids.end());
  layout.digit_prime_ids = minimax_digits(moduli, q_prime_ids, best_count);
  return layout;
}

std::optional<int64_t> maximum_modulus_bits(
    int64_t ring_dimension,
    int64_t target_bits,
    std::string_view secret_distribution,
    double error_stddev) {
  if (error_stddev != kTableErrorStddev) {
    return std::nullopt;
  }
  const bool ternary = secret_distribution == "ternary";
  if (!ternary && secret_distribution != "gaussian") {
    return std::nullopt;
  }
  for (const auto& row : kSecurityRows) {
    if (row.ternary == ternary && row.target_bits == target_bits &&
        row.ring_dimension == ring_dimension) {
      return row.maximum_bits;
    }
  }
  return std::nullopt;
}

}  // namespace fhelium::ckks
