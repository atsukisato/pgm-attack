#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pgm_utils/pgm_helpers.hpp"


inline auto inject_poisons_to_minimize_segment_length_random_adjacent(
    const std::vector<uint64_t>& data,
    std::vector<uint64_t>& poisons,
    size_t epsilon,
    size_t poison_budget,
    uint64_t seed,
    bool allow_duplicates = true
) -> size_t {
    if (data.empty()) throw std::invalid_argument("data must be non-empty");
    if (!std::is_sorted(data.begin(), data.end())) {
        throw std::invalid_argument("data must be sorted in ascending order");
    }

    poisons.clear();
    size_t n = data.size();
    size_t before_poisoning_m_opt = pgm_util::compute_m_opt(data, epsilon);

    if (poison_budget == 0) return before_poisoning_m_opt;

    std::mt19937_64 rng(seed);

    if (allow_duplicates) {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (size_t i = 0; i < poison_budget; ++i) {
            poisons.push_back(data[dist(rng)]);
        }
        std::sort(poisons.begin(), poisons.end());
    } else {
        std::vector<uint64_t> candidates;
        for (size_t i = 0; i < n; ++i) {
            if (i > 0 && data[i] > 0) {
                const uint64_t v = data[i] - 1;
                if (v > data[i - 1]) {
                    candidates.push_back(v);
                } 
            }
            if (i < n - 1 && data[i] < std::numeric_limits<uint64_t>::max()) {
                const uint64_t v = data[i] + 1;
                if (v < data[i + 1]) {
                    candidates.push_back(v);
                }
            }
        }

        if (candidates.empty()) return before_poisoning_m_opt;

        const size_t k = std::min(poison_budget, candidates.size());
        for (size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<size_t> dist(i, candidates.size() - 1);
            size_t j = dist(rng);
            std::swap(candidates[i], candidates[j]);
        }
        poisons.assign(candidates.begin(), candidates.begin() + k);
        std::sort(poisons.begin(), poisons.end());
    }

    std::vector<uint64_t> poisoned_data = data;
    poisoned_data.insert(poisoned_data.end(), poisons.begin(), poisons.end());
    std::sort(poisoned_data.begin(), poisoned_data.end());
    size_t after_poisoning_m_opt = pgm_util::compute_m_opt(poisoned_data, epsilon);
    return after_poisoning_m_opt;
}
