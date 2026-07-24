#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "minimax_regression.hpp"
#include "merge_sorted.hpp"

namespace maximize_maxerror {

/**
 * Optimal poison placement with duplicates allowed.
 * 
 * According to the theorem: In duplicate allowed setting, there exists an optimal
 * solution where all poisons are the same legitimate key value.
 * 
 * Algorithm: For each legitimate key, place lambda copies of that key as poisons,
 * compute max_error, and select the key that maximizes max_error.
 * 
 * @param keys sorted key array with sentinel removed
 * @param poisons output (clear and then store the optimal solution)
 * @param lambda number of poisons to place (same key lambda times for each candidate)
 */
inline void dup_optimal(const std::vector<uint64_t>& keys,
                        std::vector<uint64_t>& poisons,
                        size_t lambda) {
    poisons.clear();
    
    if (keys.empty()) {
        throw std::runtime_error("keys must be non-empty");
    }
    if (lambda == 0) {
        return;
    }

    // Sanity check: strictly increasing
    if (!std::is_sorted(keys.begin(), keys.end())) {
        throw std::runtime_error("keys must be sorted");
    }

    double best_max_error = -std::numeric_limits<double>::infinity();
    std::vector<uint64_t> best_poisons;
    uint64_t best_key = 0;

    const size_t n = keys.size();
    
    // For each legitimate key, try placing lambda copies of it
    for (size_t i = 0; i < n; ++i) {
        if (i > 0 && keys[i] == keys[i - 1]) {
            continue;
        }
        uint64_t candidate_key = keys[i];
        
        // Create poisons: lambda copies of candidate_key
        std::vector<uint64_t> candidate_poisons(lambda, candidate_key);
        
        // Merge keys and poisons (poisons are duplicates of a legitimate key)
        auto poisoned_keys = merge_sorted(keys, candidate_poisons);
        
        // Compute max_error using minimax regression
        auto [slope, intercept, max_error] = minimax_maxabs_regression_rank(poisoned_keys);
        (void)slope; (void)intercept;
        
        if (max_error > best_max_error) {
            best_max_error = max_error;
            best_key = candidate_key;
            best_poisons = candidate_poisons;
        }
    }

    if (best_poisons.empty()) {
        throw std::runtime_error("Failed to find optimal duplicate poison placement");
    }

    poisons = std::move(best_poisons);
}

}  // namespace maximize_maxerror
