#pragma once

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <optional>
#include <cmath>
#include <pgm/piecewise_linear_model.hpp>
#include <random>
#include <set>
#include "pgm_utils/pgm_helpers.hpp"
#include "minimize_segment_length/minimize_segment_length_swing_lambda.hpp"
#include "minimize_segment_length/minimize_segment_length_swing_lambda_with_theta.hpp"

#include "timer.hpp"


/**
 * @brief Performs linear regression on the given x and y values
 *
 * @param x X values
 * @param y Y values
 * @param slope Output slope
 * @param intercept Output intercept
 */
static inline void linear_regression_with_theta(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double& slope,
    double& intercept
) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }
    slope = (sum_xy - sum_x * sum_y / x.size()) / (sum_x2 - sum_x * sum_x / x.size());
    intercept = sum_y / x.size() - slope * sum_x / x.size();
}


/**
 * @brief Measures the theta0 parameter for the swing-lambda algorithm
 *
 * @param data Sorted input data
 * @param lambda_per_segment_candidates Lambda per segment candidates
 * @param legitimate_m_opt Legitimate m_opt
 * @param epsilon Epsilon parameter for PGM-index
 * @param theta0_measurement_trials Number of trials for theta0 measurement
 * @param allow_duplicates Whether duplicate keys in data are allowed
 * @return The theta0 parameter (average saved legitimate keys num per one poisoned key)
 */
static inline double measure_theta0_with_theta(
    const std::vector<uint64_t>& data,
    const std::vector<size_t>& lambda_per_segment_candidates,
    size_t legitimate_m_opt,
    size_t epsilon,
    size_t theta0_measurement_trials,
    bool allow_duplicates = true
) {
    double theta_estimated_from_legitimate_m_opt = (static_cast<double>(data.size()) / legitimate_m_opt - 1.0) / (2.0 * epsilon + 1.0);
    if (lambda_per_segment_candidates.size() <= 1) {
        return theta_estimated_from_legitimate_m_opt;
    }

    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    std::vector<size_t> start_pos_list;
    for (size_t i = 0; i < theta0_measurement_trials; ++i) {
        start_pos_list.push_back(dist(gen));
    }

    std::vector<double> y_sum(theta0_measurement_trials, 0.0); // legitimate keys num in segment after poisoning
    for (size_t i = 0; i < start_pos_list.size(); ++i) {
        size_t start_pos = start_pos_list[i];
        auto segment_end_no_poison = pgm_util::extend_segment_end(data, start_pos, epsilon);
        if (segment_end_no_poison == data.size() - 1) {
            continue;
        }
        std::vector<std::pair<std::vector<uint64_t>, size_t>> swing_lambda_results = minimize_segment_length_swing_lambda::compute_consecutive_multiple_poisons_minimize_segment_length_swing_lambda(
            data,
            start_pos,
            epsilon,
            lambda_per_segment_candidates,
            start_pos,
            allow_duplicates
        );
        assert(swing_lambda_results.size() == lambda_per_segment_candidates.size());
        for (size_t j = 0; j < swing_lambda_results.size(); ++j) {
            const auto& poisons = swing_lambda_results[j].first;
            const size_t segment_end_with_poisons = swing_lambda_results[j].second;
            y_sum[j] += static_cast<double>(segment_end_with_poisons - start_pos + 1);
        }
    }

    std::vector<double> lambda_per_segment_candidates_d(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end());
    std::vector<double> y_mean;
    for (size_t j = 0; j < lambda_per_segment_candidates.size(); ++j) {
        y_mean.push_back(y_sum[j] / theta0_measurement_trials);
    }

    double slope, intercept;
    linear_regression_with_theta(lambda_per_segment_candidates_d, y_mean, slope, intercept);

    if (slope <= 0.0) {
        return theta_estimated_from_legitimate_m_opt;
    }

    return -slope;
}


/**
 * @brief Algorithm to inject poisons per segment to minimize segment length using swing-lambda with theta.
 * Sets theta internally (like inject_poisons_to_minimize_segment_length_swing_lambda) and uses it to call
 * minimize_segment_length_swing_lambda_with_theta.
 *
 * @param data Sorted input data
 * @param poisons Output vector for generated poisons
 * @param epsilon Epsilon parameter for PGM-index
 * @param poison_budget Total number of poisons to inject
 * @param mu Mu parameter for theta adjustment
 * @param theta0_measurement_trials Number of trials for theta0 measurement
 * @param allow_duplicates Whether duplicate keys in data are allowed
 * @return The number of segments after poisoning
 */
inline auto inject_poisons_to_minimize_segment_length_swing_lambda_with_theta(
    const std::vector<uint64_t>& data,
    std::vector<uint64_t>& poisons,
    size_t epsilon,
    size_t poison_budget,
    double mu = 20.0,
    size_t theta0_measurement_trials = 100,
    bool allow_duplicates = true
) -> size_t {
    if (data.empty()) throw std::invalid_argument("data must be non-empty");
    if (!std::is_sorted(data.begin(), data.end())) {
        throw std::invalid_argument("data must be sorted in ascending order");
    }
    if (mu < 0.0) {
        throw std::invalid_argument("mu must be non-negative");
    }

    poisons.clear();
    size_t n = data.size();
    size_t current_i = 0;
    size_t segment_count = 0;

    size_t legitimate_m_opt = pgm_util::compute_m_opt(data, epsilon);

    if (poison_budget == 0) {
        return legitimate_m_opt;
    }

    // Generate lambda per segment candidates
    const size_t lambda_per_segment_candidate_num = 10;
    const size_t max_lambda_per_segment = 2 * epsilon + 1;
    std::vector<size_t> lambda_per_segment_candidates;
    for (size_t i = 0; i < lambda_per_segment_candidate_num; ++i) {
        lambda_per_segment_candidates.push_back(i * max_lambda_per_segment / (lambda_per_segment_candidate_num - 1));
    }
    std::sort(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end());
    lambda_per_segment_candidates.erase(std::unique(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end()), lambda_per_segment_candidates.end());
    assert(lambda_per_segment_candidates[0] == 0);

    // Measure theta0
    double theta0 = measure_theta0_with_theta(data, lambda_per_segment_candidates, legitimate_m_opt, epsilon, theta0_measurement_trials, allow_duplicates);

    // add expected average poisons num per segment to lambda_per_segment_candidates
    double average_segment_length = static_cast<double>(n) / legitimate_m_opt;
    double expected_segment_num = static_cast<double>(n + poison_budget * theta0) / average_segment_length;
    size_t expected_average_poisons_num_per_segment = static_cast<size_t>(std::llround(poison_budget / expected_segment_num));
    lambda_per_segment_candidates.push_back(expected_average_poisons_num_per_segment);
    std::sort(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end());
    lambda_per_segment_candidates.erase(std::unique(lambda_per_segment_candidates.begin(), lambda_per_segment_candidates.end()), lambda_per_segment_candidates.end());
    assert(lambda_per_segment_candidates[0] == 0);

    // Inject poisons
    while (current_i < n) {
        size_t segment_end = pgm_util::extend_segment_end(data, current_i, epsilon);
        assert(current_i <= segment_end && segment_end < n);

        if (poisons.size() < poison_budget) {
            double used_legitimate_key_ratio = static_cast<double>(current_i) / n;
            double used_poison_budget_ratio = static_cast<double>(poisons.size()) / poison_budget;
            double theta = theta0 * std::exp(-mu * (used_legitimate_key_ratio - used_poison_budget_ratio));

            while (lambda_per_segment_candidates.size() > 0 && lambda_per_segment_candidates.back() > poison_budget - poisons.size()) {
                lambda_per_segment_candidates.pop_back();
            }

            std::pair<std::vector<uint64_t>, size_t> best_result = minimize_segment_length_swing_lambda_with_theta::compute_consecutive_multiple_poisons_minimize_segment_length_swing_lambda_with_theta(
                data,
                current_i,
                epsilon,
                lambda_per_segment_candidates,
                current_i + poisons.size(),
                theta,
                allow_duplicates
            );

            if (best_result.first.size() + poisons.size() > poison_budget) {
                throw std::runtime_error("result.first.size() + poisons.size() > poison_budget");
            }

            // Add poisons to the output
            poisons.insert(poisons.end(), best_result.first.begin(), best_result.first.end());

            // Move to the next segment
            current_i = best_result.second + 1;
        } else {
            current_i = segment_end + 1;
        }

        segment_count++;
    }

    return segment_count;
}
