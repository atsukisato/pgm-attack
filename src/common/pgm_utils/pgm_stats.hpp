#pragma once
#include <cstddef>
#include <vector>
#include "pgm_utils/pgm_helpers.hpp"

namespace pgm_util {

struct IndexStats {
    size_t epsilon;
    size_t height;
    size_t bottom_segments;
    size_t size_in_bytes;
    std::vector<size_t> segments_per_level; // level 0 = top, last = bottom
};

template <typename K, size_t EPSILON>
inline IndexStats get_stats(const pgm::PGMIndex<K, EPSILON>& index) {
    IndexStats s;
    s.epsilon = EPSILON;
    s.height = index.height();
    s.bottom_segments = index.segments_count();
    s.size_in_bytes = index.size_in_bytes();

    s.segments_per_level.assign(s.height, 0);
    if (s.height > 0) {
        // The pgm-index public API exposes only the bottom-level segments count and height.
        // Populate the bottom level; other levels are left as 0 so callers can detect missing data.
        s.segments_per_level[s.height - 1] = s.bottom_segments;
    }
    return s;
}

} // namespace pgm_util

