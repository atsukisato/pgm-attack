#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace minimize_segment_length {

/**
 * Select lambda poisons from the range [min_key, max_key] in the segment randomly.
 * Apply the same logic as maximize_maxerror::random for the segment.
 *
 * @param keys sorted key array
 * @param start_pos start position of the segment
 * @param segment_end end position of the segment (inclusive)
 * @param poisons output (clear and then add)
 * @param lambda number of poisons to select
 * @param seed random seed for reproducibility
 * @param allow_duplicates true if values in keys can also be added to poisons, false if only values not in keys can be added
 */
inline void random(const std::vector<uint64_t>& keys,
                   size_t start_pos,
                   size_t segment_end,
                   std::vector<uint64_t>& poisons,
                   size_t lambda,
                   uint64_t seed,
                   bool allow_duplicates = false) {
    poisons.clear();
    if (keys.empty() || start_pos > segment_end || segment_end >= keys.size() || lambda == 0) return;

    uint64_t min_key = keys[start_pos];
    uint64_t max_key = keys[segment_end];

    const uint64_t range = (max_key - min_key) + 1;
    if (range == 0) return;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, range - 1);

    std::unordered_set<uint64_t> key_set(keys.begin() + start_pos, keys.begin() + segment_end + 1);

    if (allow_duplicates) {
        for (size_t i = 0; i < lambda; ++i) {
            uint64_t offset = dist(rng);
            poisons.push_back(min_key + offset);
        }
    } else {
        std::unordered_set<uint64_t> poisons_set;
        const size_t max_trials = lambda * 200;
        size_t trials = 0;
        while (poisons.size() < lambda && trials < max_trials) {
            uint64_t offset = dist(rng);
            uint64_t v = min_key + offset;
            if (key_set.find(v) == key_set.end() && poisons_set.find(v) == poisons_set.end()) {
                poisons_set.insert(v);
                poisons.push_back(v);
            }
            ++trials;
        }
    }
    std::sort(poisons.begin(), poisons.end());
}

}  // namespace minimize_segment_length
