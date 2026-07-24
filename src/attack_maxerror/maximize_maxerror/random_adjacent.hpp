#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

namespace maximize_maxerror {

/**
 * allow_duplicates true: Select lambda poisons from the keys randomly.
 * allow_duplicates false: Select lambda poisons from the positions adjacent to the keys (keys[i]-1 or keys[i]+1) that are not in keys and satisfy min_key < v < max_key, randomly.
 *
 * @param keys sorted key array with sentinel removed
 * @param poisons output (clear and then add)
 * @param lambda number of poisons to select
 * @param seed random seed for reproducibility
 * @param allow_duplicates true if keys only should be selected, false if keys±1 should be selected
 */
inline void random_adjacent(const std::vector<uint64_t>& keys,
                           std::vector<uint64_t>& poisons,
                           size_t lambda,
                           uint64_t seed,
                           bool allow_duplicates = false) {
    poisons.clear();
    if (keys.empty() || lambda == 0) return;

    const uint64_t min_key = keys.front();
    const uint64_t max_key = keys.back();
    std::mt19937_64 rng(seed);

    if (allow_duplicates) {
        // keys only as candidates
        std::vector<uint64_t> candidates = keys;
        if (candidates.empty()) return;

        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        for (size_t i = 0; i < lambda; ++i) {
            poisons.push_back(candidates[dist(rng)]);
        }
        std::sort(poisons.begin(), poisons.end());
    } else {
        // keys ± 1 only as candidates (not in keys and satisfy min_key < v < max_key)
        std::unordered_set<uint64_t> key_set(keys.begin(), keys.end());
        std::unordered_set<uint64_t> candidates_set;
        const size_t n = keys.size();

        for (size_t i = 0; i < n; ++i) {
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

}  // namespace maximize_maxerror
