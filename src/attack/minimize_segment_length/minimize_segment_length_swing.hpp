#ifndef MINIMIZE_SEGMENT_LENGTH_SWING_HPP
#define MINIMIZE_SEGMENT_LENGTH_SWING_HPP

#include "minimize_segment_length_swing_common.hpp"
#include "pgm_utils/pgm_helpers.hpp"

using namespace minimize_segment_length_swing_common;

constexpr size_t INTERCEPT_CANDIDATE_NUM = 41;

/**
 * @brief Finds poisons that minimize segment length for a given poison budget using a heuristic.
 *
 * @param keys Sorted input keys
 * @param start_pos Starting position of the segment
 * @param epsilon Epsilon parameter for PGM-index
 * @param poison_budget Number of poisons to inject
 * @param i_in_poisoned_data Index in poisoned data
 * @param allow_duplicates Whether duplicate keys in keys are allowed
 * @return Pair of (optimal poisons, segment end index after poisoning)
 */
std::pair<std::vector<uint64_t>, size_t> compute_consecutive_multiple_poisons_minimize_segment_length_swing(
    const std::vector<uint64_t>& keys,
    size_t start_pos,
    size_t epsilon,
    size_t poison_budget,
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
    if (keys.size() < 2 || poison_budget == 0) {
        size_t segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
        return std::make_pair(std::vector<uint64_t>{}, segment_end);
    }

    size_t initial_segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
    const std::vector<uint64_t> original_keys(keys.begin() + start_pos, keys.begin() + initial_segment_end + 1);

    double epsilon_ld = static_cast<double>(epsilon);
    std::vector<double> intercept_candidates(INTERCEPT_CANDIDATE_NUM);
    double intercept_step = (2.0 * epsilon_ld) / (INTERCEPT_CANDIDATE_NUM - 1);
    for (size_t c = 0; c < INTERCEPT_CANDIDATE_NUM; ++c) {
        intercept_candidates[c] = -epsilon_ld + c * intercept_step;
    }

    std::vector<uint64_t> poisons = compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl(
        original_keys, epsilon, poison_budget, intercept_candidates, allow_duplicates
    );
    std::sort(poisons.begin(), poisons.end());
    size_t segment_end_in_keys = pgm_util::get_segment_end_in_keys(keys, original_keys, poisons, epsilon, i_in_poisoned_data, start_pos, allow_duplicates);

    return std::make_pair(poisons, segment_end_in_keys);
}

#endif // MINIMIZE_SEGMENT_LENGTH_SWING_HPP
