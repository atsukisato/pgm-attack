#ifndef MINIMIZE_SEGMENT_LENGTH_SWING_LAMBDA_WITH_THETA_HPP
#define MINIMIZE_SEGMENT_LENGTH_SWING_LAMBDA_WITH_THETA_HPP

#include "minimize_segment_length_swing_common.hpp"
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pgm/piecewise_linear_model.hpp>
#include "pgm_utils/pgm_helpers.hpp"

namespace minimize_segment_length_swing_lambda_with_theta {

constexpr size_t INTERCEPT_CANDIDATE_NUM = 41;



// --- helpers -----------------------------------------

// pick poisons by "front packing": choose smallest missing integers starting from original_keys[0]
static inline std::vector<uint64_t> pick_front_packed_poisons(
    const std::vector<uint64_t>& original_keys,
    size_t lambda,
    bool allow_duplicates
) {
    std::vector<uint64_t> poisons;
    if (lambda == 0 || original_keys.empty()) return poisons;
    poisons.reserve(lambda);

    if (allow_duplicates) {
        poisons.resize(lambda);
        std::fill(poisons.begin(), poisons.end(), original_keys.front());
        return poisons;
    }

    uint64_t cur = original_keys.front();
    size_t i = 0;

    // ensure i points to first key >= cur
    while (i < original_keys.size() && original_keys[i] < cur) ++i;

    while (poisons.size() < lambda) {
        // skip if cur equals an existing key
        if (i < original_keys.size() && original_keys[i] == cur) {
            ++i;
            // beware overflow
            if (cur == std::numeric_limits<uint64_t>::max()) break;
            ++cur;
            continue;
        }
        poisons.push_back(cur);
        if (cur == std::numeric_limits<uint64_t>::max()) break;
        ++cur;
    }
    return poisons;
}


/**
 * @brief Calculate the score of a result
 * 
 * @param result Result
 * @param theta The theta parameter
 * @return The score
 */
 double calculate_score(
    const std::pair<std::vector<uint64_t>, size_t>& result,
    size_t current_i,
    double theta
) {
    const size_t poisons_num = result.first.size();
    const size_t key_num = result.second - current_i + 1;

    // std::cout << "key_num: " << key_num << ", poisons_num: " << poisons_num << ", theta: " << theta << std::endl;
    return static_cast<double>(key_num) + theta * static_cast<double>(poisons_num);
}

size_t get_largest_length_achieving_best_score(
    double best_score,
    double theta,
    size_t lambda_per_segment
) {
    double largest_length_achieving_best_score_d = best_score - theta * static_cast<double>(lambda_per_segment);
    if (largest_length_achieving_best_score_d < 0.0) {
        return 0;
    }
    return std::ceil(largest_length_achieving_best_score_d);
}

/**
 * @brief Finds poisons that minimize segment length for a given poison budget using a heuristic with lambda parameter.
 *
 * @param keys Sorted input keys
 * @param start_pos Starting position of the segment
 * @param epsilon Epsilon parameter for PGM-index
 * @param lambda_per_segment_candidates Number of poisons to inject (candidates)
 * @param i_in_poisoned_data Index in poisoned data
 * @param theta Theta parameter (double)
 * @param allow_duplicates Whether duplicate keys in keys are allowed
 * @return Pair of (optimal poisons, segment end index after poisoning) with minimum score (key_num + theta * poisons_num)
 */
std::pair<std::vector<uint64_t>, size_t> compute_consecutive_multiple_poisons_minimize_segment_length_swing_lambda_with_theta(
    const std::vector<uint64_t>& keys,
    size_t start_pos,
    size_t epsilon,
    const std::vector<size_t>& lambda_per_segment_candidates,
    size_t i_in_poisoned_data,
    double theta,
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
        size_t segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);
        return std::make_pair(std::vector<uint64_t>{}, segment_end);
    }
    if (!std::is_sorted(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end())) {
        throw std::invalid_argument("lambda_per_segment_candidates must be sorted");
    }

    size_t initial_segment_end = pgm_util::extend_segment_end(keys, start_pos, epsilon, i_in_poisoned_data - start_pos);

    double epsilon_ld = static_cast<double>(epsilon);
    std::vector<double> intercept_candidates(INTERCEPT_CANDIDATE_NUM);
    double intercept_step = (2.0 * epsilon_ld) / (INTERCEPT_CANDIDATE_NUM - 1);
    for (size_t c = 0; c < INTERCEPT_CANDIDATE_NUM; ++c) {
        intercept_candidates[c] = -epsilon_ld + c * intercept_step;
    }

    std::vector<bool> seen(lambda_per_segment_candidates.size(), false);
    std::pair<std::vector<uint64_t>, size_t> best_result = std::make_pair(std::vector<uint64_t>{}, initial_segment_end);
    double best_score = std::numeric_limits<double>::max();

    auto run_for_lambda_per_segment = [&](size_t lambda_per_segment_idx, size_t key_num_upper_bound) {
        if (seen[lambda_per_segment_idx]) {
            return;
        }
        seen[lambda_per_segment_idx] = true;
        size_t lambda_per_segment = lambda_per_segment_candidates[lambda_per_segment_idx];

        if (lambda_per_segment == 0) {
            std::vector<uint64_t> poisons;
            size_t segment_end_in_keys = initial_segment_end;
            std::pair<std::vector<uint64_t>, size_t> result = std::make_pair(poisons, segment_end_in_keys);
            double score = calculate_score(result, start_pos, theta);
            if (score < best_score) {
                best_result = result;
                best_score = score;
            }
            return;
        }

        // shrink the original_keys to that can achieve best_score
        std::vector<uint64_t> original_keys = std::vector<uint64_t>(keys.begin() + start_pos, keys.begin() + std::min(keys.size(), std::min(start_pos + key_num_upper_bound + 2, initial_segment_end + 2)));

        // compute poisons
        std::vector<uint64_t> poisons = minimize_segment_length_swing_common::compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl(
            original_keys, epsilon, lambda_per_segment, intercept_candidates, allow_duplicates
        );
        if (poisons.size() < lambda_per_segment) {
            return;
        }
        std::sort(poisons.begin(), poisons.end());
        size_t segment_end_in_keys = pgm_util::get_segment_end_in_keys(
            keys, poisons, epsilon, i_in_poisoned_data, start_pos, allow_duplicates
        );

        // update best_results and best_score if the current result is better
        std::pair<std::vector<uint64_t>, size_t> result = std::make_pair(poisons, segment_end_in_keys);
        double score = calculate_score(result, start_pos, theta);

        if (score < best_score) {
            best_result = result;
            best_score = score;
        }
    };


    // ------------------------------------------
    // Step 1: Compute poisons for max_lambda_per_segment to get the lower bound of score
    // ------------------------------------------
    {
        // First, try to find the largest lambda_per_segment that can be put in the segment
        std::vector<uint64_t> original_keys = std::vector<uint64_t>(keys.begin() + start_pos, keys.begin() + std::min(initial_segment_end + 2, keys.size()));
        std::vector<uint64_t> ub_poisons;
        size_t ub_poisons_lambda_per_segment_idx = 0;
        for (size_t i = 0; i < lambda_per_segment_candidates.size(); ++i) {
            size_t lambda_per_segment = lambda_per_segment_candidates[lambda_per_segment_candidates.size() - i - 1];
            std::vector<uint64_t> poisons = pick_front_packed_poisons(original_keys, lambda_per_segment, allow_duplicates);
            if (poisons.size() == 0 || poisons.back() > keys.back() || poisons.size() < lambda_per_segment) {
                // We cannot put lambda_per_segment poisons in the segment
                seen[lambda_per_segment_candidates.size() - i - 1] = true;
                continue;
            }
            ub_poisons = std::move(poisons);
            ub_poisons_lambda_per_segment_idx = lambda_per_segment_candidates.size() - i - 1;
            break;
        }
        if (ub_poisons.empty()) {
            // no valid poisons found
            std::pair<std::vector<uint64_t>, size_t> result = std::make_pair(std::vector<uint64_t>{}, initial_segment_end);
            return result;
        }
        std::sort(ub_poisons.begin(), ub_poisons.end());
        size_t segment_end_in_keys_with_ub_poisons = pgm_util::get_segment_end_in_keys(
            keys, ub_poisons, epsilon, i_in_poisoned_data, start_pos, allow_duplicates
        );
        std::pair<std::vector<uint64_t>, size_t> ub_result = std::make_pair(ub_poisons, segment_end_in_keys_with_ub_poisons);
        double ub_score = calculate_score(ub_result, start_pos, theta);
        if (ub_score < best_score) {
            best_result = ub_result;
            best_score = ub_score;
        }

        size_t key_num_upper_bound = segment_end_in_keys_with_ub_poisons - start_pos + 1;

        run_for_lambda_per_segment(ub_poisons_lambda_per_segment_idx, key_num_upper_bound);
    }

    for (size_t lambda_per_segment_idx : std::vector<size_t>{
        lambda_per_segment_candidates.size() - 1,
        lambda_per_segment_candidates.size() - 2
    }) {
        size_t lambda_per_segment = lambda_per_segment_candidates[lambda_per_segment_idx];
        size_t largest_length_achieving_best_score = get_largest_length_achieving_best_score(best_score, theta, lambda_per_segment);
        run_for_lambda_per_segment(lambda_per_segment_idx, largest_length_achieving_best_score);
    }

    for (size_t lambda_per_segment_idx = 0; lambda_per_segment_idx < lambda_per_segment_candidates.size(); ++lambda_per_segment_idx) {
        size_t lambda_per_segment = lambda_per_segment_candidates[lambda_per_segment_idx];
        size_t largest_length_achieving_best_score = get_largest_length_achieving_best_score(best_score, theta, lambda_per_segment);
        run_for_lambda_per_segment(lambda_per_segment_idx, largest_length_achieving_best_score);
    }

    return best_result;
}

} // namespace minimize_segment_length_swing_lambda_with_theta

#endif // MINIMIZE_SEGMENT_LENGTH_SWING_LAMBDA_WITH_THETA_HPP
