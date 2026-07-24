#ifndef MINIMIZE_SEGMENT_LENGTH_HPP
#define MINIMIZE_SEGMENT_LENGTH_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pgm/piecewise_linear_model.hpp>
#include "pgm_utils/pgm_helpers.hpp"


struct MinimizeSegmentLengthCandidate {
    size_t segment_end;
    std::vector<uint64_t> poisons;
    bool valid;
};


bool compare_minimize_segment_length_candidate(const MinimizeSegmentLengthCandidate& a, const MinimizeSegmentLengthCandidate& b) {
    if (!a.valid) return false;
    if (!b.valid) return true;
    if (a.segment_end != b.segment_end) return a.segment_end < b.segment_end;
    return a.poisons < b.poisons;
}


std::pair<std::vector<uint64_t>, size_t> compute_optimal_consecutive_multiple_poisons_minimize_segment_length_not_allow_duplicates(
    const std::vector<uint64_t>& keys,
    size_t start_pos,
    size_t epsilon,
    size_t poison_budget,
    size_t i_in_poisoned_data
) {
    if (keys.empty()) {
        throw std::invalid_argument("keys is empty");
    }
    if (start_pos >= keys.size()) {
        throw std::invalid_argument("start_pos is out of range");
    }
    if (i_in_poisoned_data < start_pos) {
        throw std::invalid_argument("i_in_poisoned_data is less than start_pos");
    }
    if (keys.size() < 2 || poison_budget == 0) {
        size_t segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
        return std::make_pair(std::vector<uint64_t>{}, segment_end);
    }

    size_t initial_segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
    const std::vector<uint64_t> original_keys(keys.begin() + start_pos, keys.begin() + initial_segment_end + 1);

    std::unordered_set<uint64_t> key_set(keys.begin() + start_pos, keys.begin() + initial_segment_end + 1);
    const uint64_t min_poison_value = keys[start_pos] + 1;
    const uint64_t max_poison_value = keys[initial_segment_end] - 1;

    const size_t m = initial_segment_end - start_pos + 1;
    std::vector<MinimizeSegmentLengthCandidate> best_per_i(m, MinimizeSegmentLengthCandidate{0, {}, false});

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long t = 0; t < (long long)m; ++t) {
        const size_t i = start_pos + (size_t)t;

        MinimizeSegmentLengthCandidate best{0, {}, false};

        // --- from start (right direction) ---
        if (i < initial_segment_end && keys[i] + 1 != keys[i + 1]) {
            std::vector<uint64_t> cand;
            cand.reserve(poison_budget);

            uint64_t value = keys[i] + 1;
            while (cand.size() < poison_budget && value <= max_poison_value) {
                if (key_set.find(value) == key_set.end()) cand.push_back(value);
                ++value;
            }
            if (cand.size() == poison_budget) {
                std::sort(cand.begin(), cand.end());
                size_t seg = pgm_util::get_segment_end_in_keys(keys, original_keys, cand, epsilon, i_in_poisoned_data, start_pos, false);
                best = MinimizeSegmentLengthCandidate{seg, std::move(cand), true};
            }
        }

        // --- to end (left direction) ---
        if (i > 0 && keys[i] - 1 != keys[i - 1]) {
            std::vector<uint64_t> cand;
            cand.reserve(poison_budget);

            uint64_t value = keys[i] - 1;
            while (cand.size() < poison_budget && value >= min_poison_value) {
                if (key_set.find(value) == key_set.end()) cand.push_back(value);

                if (value == min_poison_value) break;
                --value;
            }
            if (cand.size() == poison_budget) {
                std::sort(cand.begin(), cand.end());
                size_t seg = pgm_util::get_segment_end_in_keys(keys, original_keys, cand, epsilon, i_in_poisoned_data, start_pos, false);
                MinimizeSegmentLengthCandidate alt{seg, std::move(cand), true};
                if (compare_minimize_segment_length_candidate(alt, best)) best = std::move(alt);
            }
        }

        best_per_i[(size_t)t] = std::move(best);
    }

    MinimizeSegmentLengthCandidate global_best{initial_segment_end, std::vector<uint64_t>{}, true};
    for (const auto& cand : best_per_i) {
        if (compare_minimize_segment_length_candidate(cand, global_best)) global_best = cand;
    }

    if (global_best.valid) {
        return {global_best.poisons, global_best.segment_end};
    } else {
        return {std::vector<uint64_t>{}, initial_segment_end};
    }
}


std::pair<std::vector<uint64_t>, size_t> compute_optimal_consecutive_multiple_poisons_minimize_segment_length_allow_duplicates(
    const std::vector<uint64_t>& keys,
    size_t start_pos,
    size_t epsilon,
    size_t poison_budget,
    size_t i_in_poisoned_data
) {
    if (keys.empty()) {
        throw std::invalid_argument("keys is empty");
    }
    if (start_pos >= keys.size()) {
        throw std::invalid_argument("start_pos is out of range");
    }
    if (i_in_poisoned_data < start_pos) {
        throw std::invalid_argument("i_in_poisoned_data is less than start_pos");
    }
    if (keys.size() < 2 || poison_budget == 0) {
        size_t segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
        return std::make_pair(std::vector<uint64_t>{}, segment_end);
    }

    size_t initial_segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
    const std::vector<uint64_t> original_keys(keys.begin() + start_pos, keys.begin() + initial_segment_end + 1);
    const size_t m = initial_segment_end - start_pos + 1;
    std::vector<MinimizeSegmentLengthCandidate> best_per_i(m, MinimizeSegmentLengthCandidate{0, {}, false});

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
    for (long long t = 0; t < (long long)m; ++t) {
        const size_t i = start_pos + (size_t)t;
        if (t > 0 && keys[i] == keys[i - 1]) continue;

        std::vector<uint64_t> cand(poison_budget, keys[i]);
        size_t seg = pgm_util::get_segment_end_in_keys(keys, original_keys, cand, epsilon, i_in_poisoned_data, start_pos, true);
        best_per_i[(size_t)t] = MinimizeSegmentLengthCandidate{seg, std::move(cand), true};
    }

    MinimizeSegmentLengthCandidate global_best{initial_segment_end, std::vector<uint64_t>{}, true};
    for (const auto& cand : best_per_i) {
        if (compare_minimize_segment_length_candidate(cand, global_best)) global_best = cand;
    }

    if (global_best.valid) {
        return {global_best.poisons, global_best.segment_end};
    } else {
        return {std::vector<uint64_t>{}, initial_segment_end};
    }
}


/**
 * @brief Finds poisons that minimize segment length for a given poison budget.
 * 
 * @param keys Sorted input keys
 * @param start_pos Starting position of the segment
 * @param epsilon Epsilon parameter for PGM-index
 * @param poison_budget Number of poisons to inject
 * @return Pair of (optimal poisons, segment end index after poisoning)
 */
std::pair<std::vector<uint64_t>, size_t> compute_optimal_consecutive_multiple_poisons_minimize_segment_length(
    const std::vector<uint64_t>& keys,
    size_t start_pos,
    size_t epsilon,
    size_t poison_budget,
    size_t i_in_poisoned_data,
    bool allow_duplicates
) {
    if (allow_duplicates) {
        return compute_optimal_consecutive_multiple_poisons_minimize_segment_length_allow_duplicates(keys, start_pos, epsilon, poison_budget, i_in_poisoned_data);
    } else {
        return compute_optimal_consecutive_multiple_poisons_minimize_segment_length_not_allow_duplicates(keys, start_pos, epsilon, poison_budget, i_in_poisoned_data);
    }
}

#endif // MINIMIZE_SEGMENT_LENGTH_HPP
