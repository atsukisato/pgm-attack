#ifndef MINIMIZE_SEGMENT_LENGTH_SWING_LAMBDA_HPP
#define MINIMIZE_SEGMENT_LENGTH_SWING_LAMBDA_HPP

#include "minimize_segment_length_swing_common.hpp"
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pgm/piecewise_linear_model.hpp>
#include "pgm_utils/pgm_helpers.hpp"

namespace minimize_segment_length_swing_lambda {

constexpr size_t INTERCEPT_CANDIDATE_NUM = 41;


/**
 * @brief Finds poisons that minimize segment length for a given poison budget using a heuristic with lambda parameter.
 * 
 * @param keys Sorted input keys
 * @param start_pos Starting position of the segment
 * @param epsilon Epsilon parameter for PGM-index
 * @param lambda_per_segment Number of poisons to inject
 * @param i_in_poisoned_data Index in poisoned data
 * @param allow_duplicates Whether duplicate keys in keys are allowed
 * @return Pair of (optimal poisons, segment end index after poisoning)
 */
std::vector<std::pair<std::vector<uint64_t>, size_t>> compute_consecutive_multiple_poisons_minimize_segment_length_swing_lambda(
    const std::vector<uint64_t>& keys,
    size_t start_pos,
    size_t epsilon,
    const std::vector<size_t>& lambda_per_segment_candidates,
    size_t i_in_poisoned_data,
    bool allow_duplicates = true
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
    if (keys.size() < 2) {
        std::vector<std::pair<std::vector<uint64_t>, size_t>> results;
        size_t segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
        for (size_t lambda_per_segment : lambda_per_segment_candidates) {
            results.push_back(std::make_pair(std::vector<uint64_t>{}, segment_end));
        }
        return results;
    }
    if (!std::is_sorted(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end())) {
        throw std::invalid_argument("lambda_per_segment_candidates must be sorted");
    }

    size_t initial_segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
    std::vector<uint64_t> original_keys(keys.begin() + start_pos, keys.begin() + initial_segment_end + 1);

    double epsilon_ld = static_cast<double>(epsilon);
    std::vector<double> intercept_candidates(INTERCEPT_CANDIDATE_NUM);
    double intercept_step = (2.0 * epsilon_ld) / (INTERCEPT_CANDIDATE_NUM - 1);
    for (size_t c = 0; c < INTERCEPT_CANDIDATE_NUM; ++c) {
        intercept_candidates[c] = -epsilon_ld + c * intercept_step;
    }

    std::vector<std::pair<std::vector<uint64_t>, size_t>> raw_results;
    raw_results.resize(lambda_per_segment_candidates.size());

    for (long long t = 0; t < (long long)lambda_per_segment_candidates.size(); ++t) {
        size_t lambda_per_segment = lambda_per_segment_candidates[(size_t)t];
        std::vector<uint64_t> poisons = minimize_segment_length_swing_common::compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl(
            original_keys, epsilon, lambda_per_segment, intercept_candidates, allow_duplicates
        );
        std::sort(poisons.begin(), poisons.end());
        size_t segment_end_in_keys = pgm_util::get_segment_end_in_keys(
            keys, original_keys, poisons, epsilon, i_in_poisoned_data, start_pos, allow_duplicates
        );
        raw_results[(size_t)t] = {std::move(poisons), segment_end_in_keys};
        while (original_keys.size() > segment_end_in_keys - start_pos + 2) {
            original_keys.pop_back();
        }
    }

    std::vector<std::pair<std::vector<uint64_t>, size_t>> results(lambda_per_segment_candidates.size());
    std::vector<uint64_t> current_best_poisons = std::vector<uint64_t>{};
    size_t current_best_segment_end_in_keys = initial_segment_end;
    for (size_t t = 0; t < lambda_per_segment_candidates.size(); ++t) {
        const auto& [pois, seg_end] = raw_results[t];
        if (seg_end < current_best_segment_end_in_keys) {
            current_best_poisons = pois;
            current_best_segment_end_in_keys = seg_end;
        }
        results[t] = std::make_pair(current_best_poisons, current_best_segment_end_in_keys);
    }

    return results;
}

} // namespace minimize_segment_length_swing_lambda

#endif // MINIMIZE_SEGMENT_LENGTH_SWING_LAMBDA_HPP
