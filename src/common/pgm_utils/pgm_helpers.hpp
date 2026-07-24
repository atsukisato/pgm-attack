#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <pgm/pgm_index.hpp>
#include <stdexcept>
#include <vector>
#include <limits>
#include <cstdint>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/next.hpp>

using float128 = boost::multiprecision::cpp_bin_float_quad;

namespace pgm_util {

template <typename K, size_t EPSILON>
inline size_t epsilon_of(const pgm::PGMIndex<K, EPSILON>&) {
    return EPSILON;
}

template <typename K, size_t EPSILON>
inline size_t height_of(const pgm::PGMIndex<K, EPSILON>& idx) {
    return idx.height();
}

template <typename K, size_t EPSILON>
inline size_t bottom_level_segments_of(const pgm::PGMIndex<K, EPSILON>& idx) {
    return idx.segments_count();
}

template <typename K, size_t EPSILON>
inline size_t size_in_bytes_of(const pgm::PGMIndex<K, EPSILON>& idx) {
    return idx.size_in_bytes();
}

/**
 * @brief Computes the m_opt for the given keys and epsilon using the same algorithm as make_segmentation.
 * This matches the actual PGM-index segmentation logic including duplicate key handling and sentinel point.
 * @param keys The sorted keys to compute the m_opt for.
 * @param epsilon The epsilon to use for the computation.
 * @return The m_opt for the given keys and epsilon.
 */
 inline size_t compute_m_opt(
    const std::vector<std::uint64_t>& keys,
    size_t epsilon,
    size_t y_offset = 0
) {
    size_t n = keys.size();
    if (n == 0) return 0;

    using K = std::uint64_t;
    using Model = pgm::internal::OptimalPiecewiseLinearModel<K, size_t>;
    
    size_t segment_count = 0;
    std::optional<Model> model_opt;
    model_opt.emplace(epsilon);
    
    // Lambda to add a point, matching make_segmentation logic
    // When add_point fails, we output the segment and add the point to a new model
    auto add_point = [&](K x, size_t y) {
        if (!model_opt->add_point(x, y + y_offset)) {
            // std::cout << "[compute_m_opt] Failed to add point (" << x << ", " << y << ") to current segment." << std::endl;

            // Current segment is full, output it and start a new one
            segment_count++;
            // Create a new model and add the point that caused the failure
            model_opt.reset();
            model_opt.emplace(epsilon);
            if (!model_opt->add_point(x, y + y_offset)) {
                throw std::runtime_error("Failed to add point to new model");
            }
        }
    };

    // Add first point
    add_point(keys[0], 0);

    // Process keys from index 1 to n-2 (matching make_segmentation logic)
    for (size_t i = 1; i < n - 1; ++i) {
        if (keys[i] == keys[i - 1]) {
            // Duplicate key handling: at the end of a run of duplicate keys,
            // add a point for keys[i] + 1 if it's less than keys[i + 1]
            if (keys[i] + 1 < keys[i + 1]) {
                add_point(keys[i] + 1, i);
            }
        } else {
            add_point(keys[i], i);
        }
    }

    // Add last point if it's different from the previous one
    if (n >= 2 && keys[n - 1] != keys[n - 2]) {
        add_point(keys[n - 1], n - 1);
    }

    // Add sentinel point: ensure values greater than the last one are mapped to n
    add_point(keys[n - 1] + 1, n);

    // Count the final segment
    segment_count++;

    return segment_count;
}


/**
 * @brief Returns the maximum j such that keys[i]...keys[j] can be covered by one ε-segment.
 * Uses the same algorithm as make_segmentation, including duplicate key handling.
 */
inline std::size_t extend_segment_end(
    const std::vector<std::uint64_t>& keys,
    std::size_t i,
    std::size_t epsilon,
    std::size_t y_offset = 0
) {
    size_t n = keys.size();
    if (i >= n) throw std::out_of_range("Index i is out of range");
    if (i == n - 1) return n - 1; // single point segment

    using K = std::uint64_t;
    using Model = pgm::internal::OptimalPiecewiseLinearModel<K, size_t>;

    Model model_opt(epsilon);

    auto add_point = [&](K x, size_t y) {
        if (!model_opt.add_point(x, y + y_offset)) {
            return false;
        } else {
            return true;
        }
    };

    add_point(keys[i], i);

    for (size_t j = i + 1; j < n - 1; ++j) {
        if (keys[j] == keys[j - 1]) {
            if (keys[j] + 1 < keys[j + 1]) {
                if (!add_point(keys[j] + 1, j)) {
                    return j - 1;
                }
            }
        } else {
            if (!add_point(keys[j], j)) {
                return j - 1;
            }
        }
    }

    if (i < n - 1 && n >= 2 && keys[n - 1] != keys[n - 2]) {
        if (!add_point(keys[n - 1], n - 1)) {
            return n - 2;
        }
    }

    return n - 1;
}


/**
 * @brief Returns the segment end index in keys for the given poisoned data.
 * Maps segment_end_in_poisoned (index in poisoned_keys = original_keys + poisons) back to index in keys.
 * @param keys Full keys array
 * @param original_keys Subset of keys (segment range)
 * @param poisons Poisons to merge with original_keys
 * @param epsilon Epsilon parameter for PGM-index
 * @param i_in_poisoned_data Index in poisoned data
 * @param start_pos Start position in keys (original_keys = keys[start_pos..])
 * @param allow_duplicates Whether duplicate keys are allowed (affects mapping logic)
 * @return Segment end index in keys
 */
inline size_t get_segment_end_in_keys(
    const std::vector<std::uint64_t>& keys,
    const std::vector<std::uint64_t>& original_keys,
    const std::vector<std::uint64_t>& poisons,
    size_t epsilon,
    size_t i_in_poisoned_data,
    size_t start_pos,
    bool allow_duplicates = true
) {
    if (!std::is_sorted(original_keys.begin(), original_keys.end())) {
        throw std::invalid_argument("original_keys must be sorted");
    }
    if (!std::is_sorted(poisons.begin(), poisons.end())) {
        throw std::invalid_argument("poisons must be sorted");
    }

    if (!allow_duplicates) {
        std::vector<std::uint64_t> poisoned_keys;
        poisoned_keys.reserve(original_keys.size() + poisons.size());
        std::merge(original_keys.begin(), original_keys.end(), poisons.begin(), poisons.end(), std::back_inserter(poisoned_keys));
        size_t segment_end_in_poisoned = extend_segment_end(poisoned_keys, 0, epsilon, i_in_poisoned_data);

        std::uint64_t last_key_in_segment = poisoned_keys[segment_end_in_poisoned];
        size_t segment_end_in_keys = (std::lower_bound(original_keys.begin(), original_keys.end(), last_key_in_segment) - original_keys.begin()) + start_pos;
        assert(segment_end_in_keys >= start_pos && segment_end_in_keys < keys.size());
        if (keys[segment_end_in_keys] != last_key_in_segment) {
            segment_end_in_keys--;
        }
        return segment_end_in_keys;
    } else {
        std::vector<std::uint64_t> poisoned_keys;
        poisoned_keys.reserve(original_keys.size() + poisons.size());
        std::merge(original_keys.begin(), original_keys.end(), poisons.begin(), poisons.end(), std::back_inserter(poisoned_keys));
        size_t segment_end_in_poisoned = extend_segment_end(poisoned_keys, 0, epsilon, i_in_poisoned_data);

        size_t it_in_poisoned_keys = 0;
        size_t it_in_original_keys = 0;
        size_t it_in_poisons = 0;
        while (it_in_poisoned_keys < segment_end_in_poisoned) {
            if (it_in_original_keys >= original_keys.size()) {
                assert(poisoned_keys[it_in_poisoned_keys] == poisons[it_in_poisons]);
                it_in_poisoned_keys++;
                it_in_poisons++;
            } else if (it_in_poisons >= poisons.size()) {
                assert(poisoned_keys[it_in_poisoned_keys] == original_keys[it_in_original_keys]);
                it_in_poisoned_keys++;
                it_in_original_keys++;
            } else {
                if (poisoned_keys[it_in_poisoned_keys] == poisons[it_in_poisons]) {
                    it_in_poisoned_keys++;
                    it_in_poisons++;
                } else if (poisoned_keys[it_in_poisoned_keys] == original_keys[it_in_original_keys]) {
                    it_in_poisoned_keys++;
                    it_in_original_keys++;
                } else {
                    throw std::runtime_error("poisoned_keys[it_in_poisoned_keys] != poisons[it_in_poisons] && poisoned_keys[it_in_poisoned_keys] != original_keys[it_in_original_keys]");
                }
            }
        }
        return it_in_original_keys + start_pos - 1;
    }
}


/**
 * @brief Returns the segment end index in keys for the given poisoned data.
 *
 * We conceptually merge:
 *   keys[start_pos], keys[start_pos+1], ...
 * and
 *   poisons
 *
 * in ascending order, taking poison first when equal.
 *
 * We do NOT materialize the merged array. Instead, we stream points one by one
 * and stop as soon as the ε-segment can no longer be extended.
 *
 * Duplicate handling follows the same logic as extend_segment_end():
 *   for an interior merged point t,
 *     - if x[t] == x[t-1], add synthetic point (x[t]+1, y[t]) only when x[t]+1 < x[t+1]
 *     - otherwise add (x[t], y[t])
 *   for the final point of the actually processed prefix,
 *     - add it iff it is different from the previous point
 *
 * @param keys Full keys array
 * @param poisons Sorted poison keys
 * @param epsilon Epsilon parameter for PGM-index
 * @param i_in_poisoned_data poisoned-data y of keys[start_pos]
 * @param start_pos Start position in keys
 * @param allow_duplicates Whether duplicate keys are allowed
 * @return Segment end index in keys
 */
 inline size_t get_segment_end_in_keys(
    const std::vector<std::uint64_t>& keys,
    const std::vector<std::uint64_t>& poisons,
    size_t epsilon,
    size_t i_in_poisoned_data,
    size_t start_pos,
    bool allow_duplicates = true
) {
    if (!std::is_sorted(poisons.begin(), poisons.end())) {
        throw std::invalid_argument("poisons must be sorted");
    }

    using K = std::uint64_t;
    using Y = size_t;
    using Model = pgm::internal::OptimalPiecewiseLinearModel<K, Y>;

    const size_t n = keys.size();
    if (n == 0) return 0;
    if (start_pos >= n) throw std::invalid_argument("start_pos is out of range");

    struct PointState {
        K x;
        Y y;
        bool is_original;
        size_t key_idx;                  // valid iff is_original
        size_t last_original_idx_before; // last original key index before this point
        size_t last_original_idx_up_to;  // last original key index up to and including this point
    };

    Model model_opt(epsilon);

    auto add_point = [&](K x, Y y) -> bool {
        return model_opt.add_point(x, y);
    };

    // poisons that can appear after the first fixed point keys[start_pos]
    // tie goes to poison, so use lower_bound.
    size_t p = static_cast<size_t>(
        std::lower_bound(poisons.begin(), poisons.end(), keys[start_pos]) - poisons.begin()
    );
    size_t k = start_pos + 1;
    Y next_y = i_in_poisoned_data + 1;
    size_t last_original_idx_seen = start_pos;

    auto has_next_raw = [&]() -> bool {
        return p < poisons.size() || k < n;
    };

    auto peek_next_raw = [&]() -> PointState {
        if (p < poisons.size() && k < n) {
            if (poisons[p] <= keys[k]) {  // poison first on tie
                return PointState{
                    poisons[p],
                    next_y,
                    false,
                    0,
                    last_original_idx_seen,
                    last_original_idx_seen
                };
            } else {
                return PointState{
                    keys[k],
                    next_y,
                    true,
                    k,
                    last_original_idx_seen,
                    k
                };
            }
        } else if (p < poisons.size()) {
            return PointState{
                poisons[p],
                next_y,
                false,
                0,
                last_original_idx_seen,
                last_original_idx_seen
            };
        } else {
            return PointState{
                keys[k],
                next_y,
                true,
                k,
                last_original_idx_seen,
                k
            };
        }
    };

    auto consume_peeked = [&](const PointState& pt) {
        if (pt.is_original) {
            ++k;
            last_original_idx_seen = pt.key_idx;
        } else {
            ++p;
        }
        ++next_y;
    };

    PointState prev{
        keys[start_pos],
        i_in_poisoned_data,
        true,
        start_pos,
        start_pos,
        start_pos
    };

    add_point(prev.x, prev.y);

    if (!has_next_raw()) {
        return start_pos;
    }

    if (!allow_duplicates) {
        while (has_next_raw()) {
            PointState cur = peek_next_raw();
            if (!add_point(cur.x, cur.y)) {
                return cur.last_original_idx_before;
            }
            consume_peeked(cur);
            prev = cur;
        }
        return prev.last_original_idx_up_to;
    }

    // Need a 3-point window: prev, cur, nxt
    PointState cur = peek_next_raw();
    consume_peeked(cur);

    while (has_next_raw()) {
        PointState nxt = peek_next_raw();

        if (cur.x == prev.x) {
            if (cur.x != std::numeric_limits<K>::max() && cur.x + 1 < nxt.x) {
                if (!add_point(cur.x + 1, cur.y)) {
                    return cur.last_original_idx_before;
                }
            }
        } else {
            if (!add_point(cur.x, cur.y)) {
                return cur.last_original_idx_before;
            }
        }

        prev = cur;
        cur = nxt;
        consume_peeked(cur);
    }

    // Final point: same handling as extend_segment_end
    if (cur.x != prev.x) {
        if (!add_point(cur.x, cur.y)) {
            return cur.last_original_idx_before;
        }
    }

    return cur.last_original_idx_up_to;
}


/**
 * @brief Returns the regression line's slope and intercept for the given keys and epsilon using the same algorithm as make_segmentation.
 * @param keys The sorted keys to compute the regression line's slope and intercept for.
 * @param epsilon The epsilon to use for the computation.
 * @return The regression line's slope and intercept.
 */
inline std::pair<double, double> compute_regression_line(
    const std::vector<std::uint64_t>& keys,
    size_t epsilon,
    size_t y_offset = 0
) {
    size_t n = keys.size();
    if (n == 0) return {0, 0};

    using K = std::uint64_t;
    using Model = pgm::internal::OptimalPiecewiseLinearModel<K, size_t>;

    Model model_opt(epsilon);

    auto add_point = [&](K x, size_t y) {
        if (!model_opt.add_point(x, y + y_offset)) {
            return false;
        } else {
            return true;
        }
    };

    add_point(keys[0], 0);

    for (size_t j = 1; j < n - 1; ++j) {
        if (keys[j] == keys[j - 1]) {
            if (keys[j] + 1 < keys[j + 1]) {
                if (!add_point(keys[j] + 1, j)) {
                    throw std::runtime_error("Failed to add point to model");
                }
            }
        } else {
            if (!add_point(keys[j], j)) {
                throw std::runtime_error("Failed to add point to model");
            }
        }
    }

    if (n >= 2 && keys[n - 1] != keys[n - 2]) {
        if (!add_point(keys[n - 1], n - 1)) {
            throw std::runtime_error("Failed to add point to model");
        }
    }

    using CanonicalSegment = Model::CanonicalSegment;
    CanonicalSegment canonical_segment = model_opt.get_segment();

    return canonical_segment.get_floating_point_segment(keys[0]);
}

/**
 * @brief Returns the regression line's slope and intercept for the given keys and epsilon using the same algorithm as make_segmentation.
 * @param keys The sorted keys to compute the regression line's slope and intercept for.
 * @param ranks The ranks of the keys.
 * @param epsilon The epsilon to use for the computation.
 * @param y_offset The y offset to use for the computation.
 * @return The regression line's slope and intercept.
 */
inline std::pair<double, double> compute_regression_line(
    const std::vector<std::uint64_t>& keys,
    const std::vector<size_t>& ranks,
    size_t epsilon,
    size_t y_offset = 0
) {
    size_t n = keys.size();
    if (n == 0) return {0, 0};

    using K = std::uint64_t;
    using Model = pgm::internal::OptimalPiecewiseLinearModel<K, size_t>;

    Model model_opt(epsilon);

    auto add_point = [&](K x, size_t y) {
        if (!model_opt.add_point(x, y + y_offset)) {
            return false;
        } else {
            return true;
        }
    };

    add_point(keys[0], ranks[0]);

    for (size_t j = 1; j < n - 1; ++j) {
        if (keys[j] == keys[j - 1]) {
            if (keys[j] + 1 < keys[j + 1]) {
                if (!add_point(keys[j] + 1, ranks[j])) {
                    throw std::runtime_error("Failed to add point to model");
                }
            }
        } else {
            if (!add_point(keys[j], ranks[j])) {
                throw std::runtime_error("Failed to add point to model");
            }
        }
    }

    if (n >= 2 && keys[n - 1] != keys[n - 2]) {
        if (!add_point(keys[n - 1], ranks[n - 1])) {
            throw std::runtime_error("Failed to add point to model");
        }
    }

    using CanonicalSegment = Model::CanonicalSegment;
    CanonicalSegment canonical_segment = model_opt.get_segment();

    return canonical_segment.get_floating_point_segment(keys[0]);
}

} // namespace pgm_util

