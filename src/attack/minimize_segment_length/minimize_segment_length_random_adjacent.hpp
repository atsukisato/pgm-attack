#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

namespace minimize_segment_length {

/**
 * allow_duplicates true: Select lambda poisons from the keys in the segment randomly.
 * allow_duplicates false: Select lambda poisons from the positions adjacent to the keys in the segment (keys[i]-1 or keys[i]+1) that are not in keys and satisfy min_key < v < max_key, randomly.
 *
 * @param keys sorted key array
 * @param start_pos start position of the segment
 * @param segment_end end position of the segment (inclusive)
 * @param poisons output (clear and then add)
 * @param lambda number of poisons to select
 * @param seed random seed for reproducibility
 * @param allow_duplicates true if values in keys can also be added to poisons, false if only values not in keys can be added
 */
inline void random_adjacent(const std::vector<uint64_t>& keys,
                            size_t start_pos,
                            size_t segment_end,
                            std::vector<uint64_t>& poisons,
                            size_t lambda,
                            uint64_t seed,
                            bool allow_duplicates = false) {
    poisons.clear();
    if (keys.empty() || start_pos > segment_end || segment_end >= keys.size() || lambda == 0) return;

    const uint64_t min_key = keys[start_pos];
    const uint64_t max_key = keys[segment_end];
    std::mt19937_64 rng(seed);

    if (allow_duplicates) {
        std::vector<uint64_t> candidates(keys.begin() + start_pos, keys.begin() + segment_end + 1);
        if (candidates.empty()) return;

        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        for (size_t i = 0; i < lambda; ++i) {
            poisons.push_back(candidates[dist(rng)]);
        }
        std::sort(poisons.begin(), poisons.end());
    } else {
        std::unordered_set<uint64_t> key_set(keys.begin() + start_pos, keys.begin() + segment_end + 1);
        std::unordered_set<uint64_t> candidates_set;

        for (size_t i = start_pos; i <= segment_end; ++i) {
            if (keys[i] > 0) {
                const uint64_t v = keys[i] - 1;
                if (key_set.find(v) == key_set.end() && min_key < v && v < max_key) {
                    candidates_set.insert(v);
                }
            }
            if (keys[i] < std::numeric_limits<uint64_t>::max()) {
                const uint64_t v = keys[i] + 1;
                if (key_set.find(v) == key_set.end() && min_key < v && v < max_key) {
                    candidates_set.insert(v);
                }
            }
        }

        std::vector<uint64_t> candidates(candidates_set.begin(), candidates_set.end());
        if (candidates.empty()) return;

        const size_t k = std::min(lambda, candidates.size());
        if (k == candidates.size()) {
            poisons = std::move(candidates);
            std::sort(poisons.begin(), poisons.end());
            return;
        }

        for (size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<size_t> dist(i, candidates.size() - 1);
            size_t j = dist(rng);
            std::swap(candidates[i], candidates[j]);
        }
        poisons.assign(candidates.begin(), candidates.begin() + k);
        std::sort(poisons.begin(), poisons.end());
    }
}

}  // namespace minimize_segment_length
