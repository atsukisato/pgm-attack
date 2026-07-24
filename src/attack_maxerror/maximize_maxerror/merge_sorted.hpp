#pragma once

#include <cstdint>
#include <vector>

namespace maximize_maxerror {

// Merge two sorted arrays (no dedup; assumes no overlaps if you want strict uniqueness)
inline std::vector<uint64_t> merge_sorted(const std::vector<uint64_t>& a,
                                          const std::vector<uint64_t>& b) {
    std::vector<uint64_t> out;
    out.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) out.push_back(a[i++]);
        else             out.push_back(b[j++]);
    }
    while (i < a.size()) out.push_back(a[i++]);
    while (j < b.size()) out.push_back(b[j++]);
    return out;
}

}  // namespace maximize_maxerror
