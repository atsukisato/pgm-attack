#pragma once

#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include "minimax_regression.hpp"
#include "dup_optimal.hpp"

namespace maximize_maxerror {

/**
 * Brute force optimal poison placement to maximize max_error.
 * Places poisons in gaps between legitimate keys (no duplicates).
 * If allow_duplicates is true, call dup_optimal.
 *
 * @param keys sorted key array with sentinel removed
 * @param poisons output (clear and then store the optimal solution)
 * @param lambda number of poisons to place
 * @param allow_duplicates true if dup_optimal should be called
 */
inline void optimal(const std::vector<uint64_t>& keys,
                    std::vector<uint64_t>& poisons,
                    size_t lambda,
                    bool allow_duplicates = false) {
    poisons.clear();

    if (allow_duplicates) {
        dup_optimal(keys, poisons, lambda);
        return;
    }

    if (keys.empty()) {
        throw std::runtime_error("Empty keys");
    }
    if (lambda == 0) {
        return;
    }
    if (keys.size() < 2) {
        throw std::runtime_error("Need at least two points to place poisons between them");
    }
    if (!std::is_sorted(keys.begin(), keys.end())) {
        throw std::runtime_error("Keys must be sorted");
    }

    const size_t n = keys.size();
    std::vector<uint64_t> xs(n);
    for (size_t i = 0; i < n; ++i) {
        xs[i] = keys[i];
    }

    // Calculate capacity of each gap (integer points)
    std::vector<size_t> cap(n - 1);
    size_t total_cap = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        // xs[i]+1, ..., xs[i+1]-1 can be placed
        if (xs[i + 1] <= xs[i] + 1) {
            cap[i] = 0;  // no space in this gap
        } else {
            cap[i] = static_cast<size_t>(xs[i + 1] - xs[i] - 1);
        }
        total_cap += cap[i];
    }

    if (total_cap < lambda) {
        throw std::runtime_error("Not enough capacity to place " + std::to_string(lambda) + " poisons");
    }

    // Enumerate all (a_i, b_i) by DFS
    // a_i: number to place after xs[i], b_i: number to place before xs[i+1]
    std::vector<size_t> a(n - 1, 0), b(n - 1, 0);

    double best_max_error = -std::numeric_limits<double>::infinity();
    std::vector<uint64_t> best_poisons;

    std::function<void(size_t, size_t)> dfs = [&](size_t idx, size_t remaining) {
        if (idx == n - 1) {
            // Assignments are determined, so build new_xs and poisons
            std::vector<uint64_t> new_xs;
            new_xs.reserve(n + lambda);
            std::vector<uint64_t> candidate_poisons;
            candidate_poisons.reserve(lambda);

            for (size_t i = 0; i < n; ++i) {
                if (i > 0) {
                    // b[i-1] numbers (fill left side with consecutive integers) before xs[i]
                    const size_t bi = b[i - 1];
                    for (size_t j = 0; j < bi; ++j) {
                        uint64_t val = xs[i] - static_cast<uint64_t>(bi - j);
                        new_xs.push_back(val);
                        candidate_poisons.push_back(val);
                    }
                }

                // Original point
                new_xs.push_back(xs[i]);

                if (i < n - 1) {
                    // a[i] numbers (fill right side with consecutive integers) after xs[i]
                    const size_t ai = a[i];
                    for (size_t j = 1; j <= ai; ++j) {
                        uint64_t val = xs[i] + static_cast<uint64_t>(j);
                        new_xs.push_back(val);
                        candidate_poisons.push_back(val);
                    }
                }
            }

            // Sort merged keys
            std::sort(new_xs.begin(), new_xs.end());

            // Calculate max_error using minimax regression
            auto result = minimax_maxabs_regression_rank(new_xs);
            double max_error = result.max_error;

            if (max_error > best_max_error) {
                best_max_error = max_error;
                best_poisons = std::move(candidate_poisons);
            }
            return;
        }

        const size_t c = cap[idx];
        const size_t a_max = std::min(c, remaining);
        for (size_t ai = 0; ai <= a_max; ++ai) {
            const size_t rem_after_a = remaining - ai;
            const size_t b_max = std::min(c - ai, rem_after_a);
            for (size_t bi = 0; bi <= b_max; ++bi) {
                a[idx] = ai;
                b[idx] = bi;

                // Constraint: avoid placing poisons on both sides of consecutive gaps
                if (idx > 0 && b[idx - 1] > 0 && a[idx] > 0) {
                    continue;
                }

                dfs(idx + 1, rem_after_a - bi);
            }
        }
    };

    dfs(0, lambda);

    // Sort best poisons (should be already sorted, but just in case)
    std::sort(best_poisons.begin(), best_poisons.end());
    poisons = std::move(best_poisons);
}

}  // namespace maximize_maxerror
