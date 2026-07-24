#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "optimal.hpp"

namespace maximize_maxerror {

/**
 * Greedy optimal poison placement: add the best single poison.
 * 
 * Algorithm: Repeat lambda times:
 *   1. Call optimal() with lambda=1 to find the best single poison placement
 *   2. Add that poison to the current keys
 *   3. Repeat
 * 
 * @param keys sorted key array with sentinel removed
 * @param poisons output (clear and then store the optimal solution)
 * @param lambda number of poisons to place
 */
inline void greedy(const std::vector<uint64_t>& keys,
                   std::vector<uint64_t>& poisons,
                   size_t lambda,
                   bool allow_duplicates = false) {
    poisons.clear();
    
    if (keys.empty()) {
        throw std::runtime_error("keys must be non-empty");
    }
    if (lambda == 0) {
        return;
    }

    // Sanity check: strictly increasing
    if (allow_duplicates) {
        if (!std::is_sorted(keys.begin(), keys.end())) {
            throw std::runtime_error("keys must be sorted");
        }
    } else {
        for (size_t k = 1; k < keys.size(); ++k) {
            if (!(keys[k - 1] < keys[k])) {
                throw std::runtime_error("keys must be strictly increasing (sorted unique)");
            }
        }
    }

    // Current state: keys + poisons (sorted)
    std::vector<uint64_t> current_keys = keys;

    // Greedy iteration: add one poison at a time using optimal(lambda=1)
    for (size_t iter = 0; iter < lambda; ++iter) {
        // Use optimal() to find the best single poison placement
        std::vector<uint64_t> single_poison;
        
        try {
            optimal(current_keys, single_poison, 1, allow_duplicates);
        } catch (const std::runtime_error&) {
            // If optimal() fails, break the loop
            break;
        }

        if (single_poison.empty()) {
            break;
        }

        // Add the best poison
        poisons.push_back(single_poison[0]);
        current_keys.push_back(single_poison[0]);
        std::sort(current_keys.begin(), current_keys.end());
    }

    std::sort(poisons.begin(), poisons.end());
}

}  // namespace maximize_maxerror
