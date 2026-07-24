#pragma once

#include <cstdint>
#include <vector>

#include "pgm_utils/pgm_helpers.hpp"

namespace upper_bound_utils {

/**
 * Return the number of blocks when the key array is divided into blocks with epsilon tolerance.
 * Equivalent to size - 1 of the result of devide_into_blocks. Efficient when indices are not needed.
 *
 * @param keys sorted key array
 * @param epsilon PGM epsilon parameter
 * @return number of blocks
 */
inline size_t count_blocks(const std::vector<uint64_t>& keys, size_t epsilon) {
    size_t count = 0;
    size_t current_i = 0;
    while (current_i < keys.size()) {
        ++count;
        size_t segment_end = pgm_util::extend_segment_end(keys, current_i, epsilon, 1'000'000'000);
        current_i = segment_end + 1;
    }
    return count;
}

/**
 * Divide the key array into blocks with epsilon tolerance.
 * Each block is the maximum interval that can be covered by one segment of PGM.
 *
 * @param keys sorted key array
 * @param epsilon PGM epsilon parameter
 * @return the starting index of each block (including keys.size() at the end)
 */
inline std::vector<size_t> devide_into_blocks(const std::vector<uint64_t>& keys, size_t epsilon) {
    std::vector<size_t> block_indices;
    size_t current_i = 0;
    while (current_i < keys.size()) {
        block_indices.push_back(current_i);
        size_t segment_end = pgm_util::extend_segment_end(keys, current_i, epsilon, 1'000'000'000);
        current_i = segment_end + 1;
    }
    block_indices.push_back(keys.size());
    return block_indices;
}

}  // namespace upper_bound_utils
