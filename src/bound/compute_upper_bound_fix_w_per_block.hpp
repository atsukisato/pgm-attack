#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "upper_bound_utils/compute_D_nu_o_nlogn.hpp"
#include "upper_bound_utils/devide_into_blocks.hpp"
#include "upper_bound_utils/maximize_seg_sum_mckp_concave_greedy_implicit.hpp"
#include "pgm_utils/pgm_helpers.hpp"

namespace upper_bound_fix_w_per_block {

/** 1.0 to 5.0 in 1.0 increments, then 2x increments up to two_epsilon. Exclude values greater than 2*epsilon. */
inline std::vector<double> make_nu_values(size_t epsilon) {
    const double two_epsilon = static_cast<double>(2 * epsilon);
    std::vector<double> nu_values;
    for (double nu = 1.0; nu <= 5.0; nu += 1.0) {
        if (nu <= two_epsilon) nu_values.push_back(nu);
    }
    for (double nu = 10.0; nu <= two_epsilon; nu *= 2.0) {
        if (nu <= two_epsilon) nu_values.push_back(nu);
    }
    return nu_values;
}

/** List of epsilon values to try for block division */
inline std::vector<size_t> make_epsilons_to_try(size_t epsilon) {
    return {
        epsilon,
        static_cast<size_t>(epsilon * 1.2),
        static_cast<size_t>(epsilon * 1.4),
        static_cast<size_t>(epsilon * 1.6),
        static_cast<size_t>(epsilon * 1.8),
        epsilon * 2
    };
}

/** Compute the upper bound information for one block */
inline upper_bound_utils::BlockUpperBoundImplicit make_block_upper_bound(
    const std::vector<uint64_t>& block_keys, size_t epsilon,
    size_t epsilon_to_devide_into_blocks, const std::vector<double>& nu_values) {

    std::vector<uint64_t> reduced_keys, reduced_ranks;
    {
        size_t n = block_keys.size();
        reduced_keys.push_back(block_keys[0]);
        reduced_ranks.push_back(0);
        for (size_t i = 1; i < n - 1; ++i) {
            if (block_keys[i] == block_keys[i - 1]) {
                if (block_keys[i] + 1 < block_keys[i + 1]) {
                    reduced_keys.push_back(block_keys[i] + 1);
                    reduced_ranks.push_back(i);
                }
            } else {
                reduced_keys.push_back(block_keys[i]);
                reduced_ranks.push_back(i);
            }
        }
        if (n >= 2 && block_keys[n - 1] != block_keys[n - 2]) {
            reduced_keys.push_back(block_keys[n - 1]);
            reduced_ranks.push_back(n - 1);
        }
    }

    auto [slope, _] = pgm_util::compute_regression_line(reduced_keys, reduced_ranks, epsilon_to_devide_into_blocks, 1'000'000'000);

    std::vector<double> A(reduced_keys.size());
    for (size_t i = 0; i < reduced_keys.size(); ++i) {
        A[i] = slope * static_cast<double>(reduced_keys[i]) - static_cast<double>(reduced_ranks[i]);
    }

    size_t segment_num_in_block = upper_bound_utils::count_blocks(block_keys, epsilon);
    std::vector<double> D(nu_values.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long j_ll = 0; j_ll < (long long)nu_values.size(); ++j_ll) {
        const size_t j = (size_t)j_ll;
        if (nu_values[j] == 1.0) {
            D[j] = -static_cast<double>(segment_num_in_block);
        } else {
            D[j] = upper_bound_utils::compute_D_nu_o_nlogn(A, nu_values[j], epsilon);
        }
    }

    upper_bound_utils::BlockUpperBoundImplicit blk;
    blk.nu_values = &nu_values;
    blk.D = std::move(D);
    return blk;
}

/** Compute the upper bound information for all blocks */
inline std::vector<upper_bound_utils::BlockUpperBoundImplicit> make_blocks(
    const std::vector<uint64_t>& keys, const std::vector<size_t>& block_indices, size_t epsilon,
    size_t epsilon_to_devide_into_blocks, const std::vector<double>& nu_values) {
    const size_t block_num = block_indices.size() - 1;
    std::vector<upper_bound_utils::BlockUpperBoundImplicit> blocks;
    blocks.reserve(block_num);

    for (size_t i = 0; i < block_num; ++i) {
        std::vector<uint64_t> block_keys(keys.begin() + block_indices[i],
                                         keys.begin() + block_indices[i + 1]);
        uint64_t block_begin_key = block_keys[0];
        for (size_t j = 0; j < block_keys.size(); ++j) block_keys[j] -= block_begin_key;

        blocks.push_back(make_block_upper_bound(block_keys, epsilon, epsilon_to_devide_into_blocks, nu_values));
    }
    return blocks;
}

/**
 * Compute the upper bound of m_opt after poisoning using the fix_w_per_block method.
 * Try multiple epsilon values for block division and return the most stringent (smallest) upper bound.
 *
 * @param keys sorted key array
 * @param epsilon PGM epsilon
 * @param lambda_all upper bound of the number of poisons (total λ)
 * @return upper bound of m_opt
 */
inline size_t compute_m_opt_after_poisoning_upper_bound(
    const std::vector<uint64_t>& keys, size_t epsilon, size_t lambda_all) {

    // check if it is a single value
    {
        std::vector<uint64_t> unique_keys = keys;
        std::sort(unique_keys.begin(), unique_keys.end());
        unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());
        std::sort(unique_keys.begin(), unique_keys.end());
        if (unique_keys.size() <= 1) return 2;
        if (unique_keys.size() == 2 && unique_keys[1] - unique_keys[0] == 1) return 2;
    }

    const std::vector<double> nu_values = make_nu_values(epsilon);
    const std::vector<size_t> epsilons_to_try = make_epsilons_to_try(epsilon);

    size_t m_opt_after_poisoning_upper_bound = std::numeric_limits<size_t>::max();

    for (size_t epsilon_to_devide_into_blocks : epsilons_to_try) {
        std::vector<size_t> block_indices = upper_bound_utils::devide_into_blocks(keys, epsilon_to_devide_into_blocks);
        std::vector<upper_bound_utils::BlockUpperBoundImplicit> blocks = make_blocks(keys, block_indices, epsilon, epsilon_to_devide_into_blocks, nu_values);
        const double max_seg_sum_mckp = upper_bound_utils::maximize_seg_sum_mckp_concave_greedy_implicit(blocks, lambda_all);
        const size_t max_seg_sum = static_cast<size_t>(std::ceil(max_seg_sum_mckp));
        m_opt_after_poisoning_upper_bound = std::min(m_opt_after_poisoning_upper_bound, max_seg_sum);
        if (blocks.size() == 1) {
            break;
        }
    }

    // obvious upper bound (range)
    {
        auto [mi, ma] = std::minmax_element(keys.begin(), keys.end());
        uint64_t range = *ma - *mi;
        size_t obvious_upper_bound_range = range / 2 + 1;
        m_opt_after_poisoning_upper_bound = std::min(m_opt_after_poisoning_upper_bound, obvious_upper_bound_range);
    }

    return m_opt_after_poisoning_upper_bound;
}

}  // namespace upper_bound_fix_w_per_block
