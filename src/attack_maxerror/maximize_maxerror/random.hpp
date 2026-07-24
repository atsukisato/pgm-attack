#pragma once

#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace maximize_maxerror {

/**
 * Select lambda poisons from the range [min_key, max_key] randomly and store in poisons.
 * Do not create a candidate vector, and select the offset from [0, max_key-min_key] randomly to achieve O(lambda) memory.
 *
 * @param keys sorted key array with sentinel removed
 * @param poisons output (clear and then add)
 * @param lambda number of poisons to select
 * @param seed random seed for reproducibility
 * @param allow_duplicates true if values in keys can also be added to poisons, false if only values not in keys can be added
 */
inline void random(const std::vector<uint64_t>& keys,
                   std::vector<uint64_t>& poisons,
                   size_t lambda,
                   uint64_t seed,
                   bool allow_duplicates = false) {
    poisons.clear();
    if (keys.empty() || lambda == 0) return;

    uint64_t min_key = keys.front();
    uint64_t max_key = keys.back();

    const uint64_t range = (max_key - min_key) + 1;
    if (range == 0) return;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, range - 1);

    if (allow_duplicates) {
        // values in keys are also OK → sample lambda values uniformly from the range
        for (size_t i = 0; i < lambda; ++i) {
            uint64_t offset = dist(rng);
            poisons.push_back(min_key + offset);
        }
    } else {
        std::unordered_set<uint64_t> key_set(keys.begin(), keys.end());
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
}

}  // namespace maximize_maxerror
