#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "pgm_utils/pgm_helpers.hpp"


inline std::vector<uint64_t> sample_without_replacement(
    uint64_t l,
    uint64_t u,
    uint64_t num,
    uint64_t seed
) {
    if (l > u) {
        throw std::invalid_argument("l must be <= u");
    }

    if (l == 0 && u == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("range size 2^64 is not supported");
    }

    const uint64_t N = u - l + 1;

    if (num > N) {
        throw std::invalid_argument("num must be <= (u - l + 1)");
    }

    std::mt19937_64 rng(seed);

    std::unordered_map<uint64_t, uint64_t> mp;
    mp.reserve(static_cast<size_t>(num));

    std::vector<uint64_t> result;
    result.reserve(static_cast<size_t>(num));

    auto get_val = [&](uint64_t idx) -> uint64_t {
        auto it = mp.find(idx);
        return (it == mp.end() ? idx : it->second);
    };

    for (uint64_t i = 0; i < num; ++i) {
        std::uniform_int_distribution<uint64_t> dist(i, N - 1);
        uint64_t j = dist(rng);

        uint64_t vi = get_val(i);
        uint64_t vj = get_val(j);

        mp[i] = vj;
        mp[j] = vi;

        result.push_back(vj + l);
    }

    return result;
}


inline auto inject_poisons_to_minimize_segment_length_random(
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

    if (allow_duplicates) {
        const uint64_t min_key = data.front();
        const uint64_t max_key = data.back();
        const uint64_t range = (max_key - min_key) + 1;
        if (range == 0) return before_poisoning_m_opt;

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, range - 1);
        for (size_t i = 0; i < poison_budget; ++i) {
            uint64_t offset = dist(rng);
            poisons.push_back(min_key + offset);
        }
        std::sort(poisons.begin(), poisons.end());

        std::vector<uint64_t> poisoned_data = data;
        poisoned_data.insert(poisoned_data.end(), poisons.begin(), poisons.end());
        std::sort(poisoned_data.begin(), poisoned_data.end());
        size_t after_poisoning_m_opt = pgm_util::compute_m_opt(poisoned_data, epsilon);
        return after_poisoning_m_opt;
    } else {
        const uint64_t min_key = data.front();
        const uint64_t max_key = data.back();
        std::vector<uint64_t> gaps(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            gaps[i] = data[i + 1] - data[i] - 1;
        }
        std::vector<uint64_t> cumulative_gaps(n);
        cumulative_gaps[0] = 0;
        for (size_t i = 1; i < n; ++i) {
            cumulative_gaps[i] = cumulative_gaps[i - 1] + gaps[i - 1];
        }

        const uint64_t total_gap_slots = cumulative_gaps.back();
        if (total_gap_slots < poison_budget) {
            poison_budget = static_cast<size_t>(total_gap_slots);
        }

        std::vector<uint64_t> sampled_offsets =
            (poison_budget == 0)
                ? std::vector<uint64_t>()
                : sample_without_replacement(0, total_gap_slots - 1, poison_budget, seed);
        std::sort(sampled_offsets.begin(), sampled_offsets.end());

        size_t gap_index = 0;
        for (size_t i = 0; i < poison_budget; ++i) {
            while (gap_index < n - 1 && sampled_offsets[i] >= cumulative_gaps[gap_index + 1]) {
                ++gap_index;
            }
            const uint64_t local = sampled_offsets[i] - cumulative_gaps[gap_index];
            poisons.push_back(data[gap_index] + local + 1);
        }

        std::sort(poisons.begin(), poisons.end());
        std::vector<uint64_t> poisoned_data = data;
        poisoned_data.insert(poisoned_data.end(), poisons.begin(), poisons.end());
        std::sort(poisoned_data.begin(), poisoned_data.end());
        size_t after_poisoning_m_opt = pgm_util::compute_m_opt(poisoned_data, epsilon);
        return after_poisoning_m_opt;
    }
}
