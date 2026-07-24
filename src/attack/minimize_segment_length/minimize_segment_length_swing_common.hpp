#ifndef MINIMIZE_SEGMENT_LENGTH_SWING_COMMON_HPP
#define MINIMIZE_SEGMENT_LENGTH_SWING_COMMON_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>
#include <optional>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <numeric>
#include <chrono>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../attack_maxerror/maximize_maxerror/minimax_regression.hpp"

namespace minimize_segment_length_swing_common {

struct SkipSetStepper {
    std::vector<uint64_t> k;      // sorted, strictly increasing
    std::vector<uint64_t> pref;   // pref[t] = sum gap[0..t-1], size = k.size()

    explicit SkipSetStepper(std::vector<uint64_t> forbidden) : k(std::move(forbidden)) {
        if (!std::is_sorted(k.begin(), k.end()))
            throw std::invalid_argument("k must be sorted");
        // for (size_t i = 1; i < k.size(); ++i) {
        //     if (k[i] <= k[i - 1])
        //         throw std::invalid_argument("k must be strictly increasing (all distinct)");
        // }

        const size_t n = k.size();
        pref.assign(n, 0);
        for (size_t i = 0; i + 1 < n; ++i) {
            uint64_t gap = 0;
            if (k[i + 1] > k[i] + 1) gap = k[i + 1] - k[i] - 1;
            pref[i + 1] = pref[i] + gap;
        }
    }

    // returns nullopt if the answer would be > k.back()
    std::optional<uint64_t> last_right(uint64_t x, uint64_t lambda) const {
        if (lambda == 0) return x;
        const size_t n = k.size();
        if (n == 0) return std::nullopt; // boundary undefined

        if (x >= k.back()) return std::nullopt;

        size_t p = std::upper_bound(k.begin(), k.end(), x) - k.begin();
        if (p == n) return std::nullopt; // would go beyond k.back()

        uint64_t len0 = k[p] - x - 1;
        if (lambda <= len0) {
            uint64_t ans = x + lambda;
            if (ans > k.back()) return std::nullopt;
            return ans;
        }

        lambda -= len0;

        uint64_t sum_rem = pref[n - 1] - pref[p];
        if (lambda > sum_rem) return std::nullopt; // would exceed k.back()

        uint64_t target = pref[p] + lambda;
        size_t t = std::lower_bound(pref.begin(), pref.end(), target) - pref.begin();

        uint64_t used_before = pref[t - 1] - pref[p];
        uint64_t off = lambda - used_before; // 1-based
        uint64_t ans = k[t - 1] + off;
        if (ans > k.back()) return std::nullopt;
        return ans;
    }

    // returns nullopt if the answer would be < k.front()
    std::optional<uint64_t> last_left(uint64_t x, uint64_t lambda) const {
        if (lambda == 0) return x;
        const size_t n = k.size();
        if (n == 0) return std::nullopt; // boundary undefined

        if (x <= k.front()) return std::nullopt;

        size_t lb = std::lower_bound(k.begin(), k.end(), x) - k.begin();
        if (lb == 0) return std::nullopt;
        size_t q = lb - 1; // last forbidden < x

        uint64_t len0 = x - k[q] - 1;
        if (lambda <= len0) {
            if (x < lambda) return std::nullopt; // underflow guard
            uint64_t ans = x - lambda;
            if (ans < k.front()) return std::nullopt;
            return ans;
        }

        lambda -= len0;

        uint64_t sum_left = pref[q]; // sum gap[0..q-1]
        if (lambda > sum_left) return std::nullopt; // would go below k.front()

        uint64_t value = pref[q] - lambda;
        size_t i = (std::upper_bound(pref.begin(), pref.begin() + (q + 1), value) - pref.begin()) - 1;

        uint64_t used_after = pref[q] - pref[i + 1];
        uint64_t r = lambda - used_after; // 1-based from right end
        uint64_t ans = k[i + 1] - r;
        if (ans < k.front()) return std::nullopt;
        return ans;
    }
};


class SparseTableMinLD {
public:
    using ld = double;

    SparseTableMinLD() = default;
    explicit SparseTableMinLD(const std::vector<ld>& a) { build(a); }

    void build(const std::vector<ld>& a) {
        n_ = (int)a.size();
        if (n_ == 0) {
            lg_.clear();
            st_.clear();
            return;
        }

        // floor(log2(i)) for i=1..n
        lg_.assign(n_ + 1, 0);
        for (int i = 2; i <= n_; ++i) lg_[i] = lg_[i / 2] + 1;

        int K = lg_[n_] + 1;
        K_ = K;
        st_.assign((size_t)K_ * (size_t)n_, std::numeric_limits<ld>::max());

        // k=0
        for (int i = 0; i < n_; ++i) {
            st_[(size_t)0 * n_ + (size_t)i] = a[i];
        }

        // build
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            int limit = n_ - len + 1;
            for (int i = 0; i < limit; ++i) {
                const ld left  = st_[(size_t)(k - 1) * n_ + (size_t)i];
                const ld right = st_[(size_t)(k - 1) * n_ + (size_t)(i + half)];
                st_[(size_t)k * n_ + (size_t)i] = std::min(left, right);
            }
        }
    }

    // inclusive range [l, r]
    ld range_min(int l, int r) const {
        if (l > r) return std::numeric_limits<ld>::max();
        check_range(l, r);
        int len = r - l + 1;
        int k = lg_[len];
        int shift = len - (1 << k);
        const ld a = st_[(size_t)k * n_ + (size_t)l];
        const ld b = st_[(size_t)k * n_ + (size_t)(l + shift)];
        return std::min(a, b);
    }

    int size() const { return n_; }

private:
    int n_ = 0;
    std::vector<int> lg_;
    int K_ = 0;
    std::vector<ld> st_;

    void check_range(int l, int r) const {
        if (n_ == 0) throw std::runtime_error("SparseTable is empty.");
        if (l < 0 || r < 0 || l >= n_ || r >= n_)
            throw std::out_of_range("Invalid range [l, r].");
    }
};


class SparseTableMaxLD {
public:
    using ld = double;

    SparseTableMaxLD() = default;
    explicit SparseTableMaxLD(const std::vector<ld>& a) { build(a); }

    void build(const std::vector<ld>& a) {
        n_ = (int)a.size();
        if (n_ == 0) {
            lg_.clear();
            st_.clear();
            return;
        }

        // floor(log2(i)) for i=1..n
        lg_.assign(n_ + 1, 0);
        for (int i = 2; i <= n_; ++i) lg_[i] = lg_[i / 2] + 1;

        int K = lg_[n_] + 1;
        K_ = K;
        st_.assign((size_t)K_ * (size_t)n_, std::numeric_limits<ld>::lowest());

        // k=0
        for (int i = 0; i < n_; ++i) {
            st_[(size_t)0 * n_ + (size_t)i] = a[i];
        }

        // build
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            int limit = n_ - len + 1;
            for (int i = 0; i < limit; ++i) {
                const ld left  = st_[(size_t)(k - 1) * n_ + (size_t)i];
                const ld right = st_[(size_t)(k - 1) * n_ + (size_t)(i + half)];
                st_[(size_t)k * n_ + (size_t)i] = std::max(left, right);
            }
        }
    }

    // inclusive range [l, r]
    ld range_max(int l, int r) const {
        if (l > r) return std::numeric_limits<ld>::min();
        check_range(l, r);
        int len = r - l + 1;
        int k = lg_[len];
        int shift = len - (1 << k);
        const ld a = st_[(size_t)k * n_ + (size_t)l];
        const ld b = st_[(size_t)k * n_ + (size_t)(l + shift)];
        return std::max(a, b);
    }

    int size() const { return n_; }

private:
    int n_ = 0;
    std::vector<int> lg_;
    int K_ = 0;
    std::vector<ld> st_;

    void check_range(int l, int r) const {
        if (n_ == 0) throw std::runtime_error("SparseTable is empty.");
        if (l < 0 || r < 0 || l >= n_ || r >= n_)
            throw std::out_of_range("Invalid range [l, r].");
    }
};


struct SlopeRange {
    double min_slope;
    double max_slope;

    bool is_feasible() const {
        return min_slope <= max_slope;
    }
};


inline SlopeRange intersect(const SlopeRange& a, const SlopeRange& b) {
    return SlopeRange{std::max(a.min_slope, b.min_slope), std::min(a.max_slope, b.max_slope)};
}


struct SwingPerICandidate {
    size_t segment_end_index_in_original_keys;
    size_t i;
    bool is_right_of_key_i;
    bool valid;
};


inline void _update_optimal_poisons(
    size_t& optimal_i,
    bool& optimal_is_right_of_key_i,
    size_t& optimal_segment_end_index_in_original_keys,
    size_t candidate_i,
    bool candidate_is_right_of_key_i,
    size_t candidate_segment_end_index_in_original_keys
) {
    if (candidate_segment_end_index_in_original_keys < optimal_segment_end_index_in_original_keys) {
        optimal_i = candidate_i;
        optimal_is_right_of_key_i = candidate_is_right_of_key_i;
        optimal_segment_end_index_in_original_keys = candidate_segment_end_index_in_original_keys;
    }
}


inline std::vector<uint64_t> compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl_with_sparse_table(
    const std::vector<uint64_t>& keys,
    size_t epsilon,
    size_t poison_budget,
    std::vector<double> intercept_at_key0_candidates,
    const std::vector<double>& pref_max_min_slope,
    const std::vector<double>& pref_min_max_slope,
    const std::vector<double>& suff_max_min_slope_right,
    const std::vector<double>& suff_min_max_slope_right,
    size_t n,
    const std::vector<SparseTableMaxLD>& st_min_slope_right_of_poisons_for_each_intercept,
    const std::vector<SparseTableMinLD>& st_max_slope_right_of_poisons_for_each_intercept,
    const SkipSetStepper& skipset_stepper
) {
    auto idx = [n](size_t c, size_t i) -> size_t { return c * n + i; };
    const size_t C = intercept_at_key0_candidates.size();
    const double epsilon_ld = static_cast<double>(epsilon);
    const double poison_budget_ld = static_cast<double>(poison_budget);

    for (const auto& intercept_at_key0 : intercept_at_key0_candidates) {
        if (intercept_at_key0 < -epsilon_ld || epsilon_ld < intercept_at_key0) {
            throw std::invalid_argument("intercept_at_key0 is out of range");
        }
    }

    size_t optimal_i = n;
    bool optimal_is_right_of_key_i = false;
    size_t optimal_segment_end_index_in_original_keys = n - 1;

    auto get_best_per_i = [&](size_t i, const std::vector<size_t>& c_indices) {
        size_t best_seg = n;
        bool best_is_right = false;
        bool best_valid = false;

        // ------------------------------------------------------------
        // Returns argmax index in [0, m-1] and its value.
        // Assumption: f is unimodal on the order of c_indices.
        // ------------------------------------------------------------
        auto argmax_unimodal_on_c_indices = [&](auto&& f) -> std::pair<size_t, size_t> {
            const size_t m = c_indices.size();
            if (m == 0) return {0, 0};
            std::vector<char> seen(m, 0);
            std::vector<size_t> val(m, 0);
            auto getv = [&](size_t t) -> size_t {
                if (!seen[t]) {
                    seen[t] = 1;
                    val[t] = f(c_indices[t]);
                }
                return val[t];
            };
            if (m == 1) return {0, getv(0)};
            const size_t STEP = std::max(static_cast<size_t>(2), static_cast<size_t>(std::sqrt(m / 2)));
            // 1) coarse scan: every 4th
            size_t t0 = 0;
            size_t t1 = 0;
            size_t best_v = getv(0);
            for (size_t t = 0; t < m; t += STEP) {
                size_t v = getv(t);
                if (v > best_v) { best_v = v; t0 = t; t1 = t; }
                else if (v == best_v) { t1 = t; }
            }
            // also check last (often missed by step)
            {
                size_t v = getv(m - 1);
                if (v > best_v) { best_v = v; t0 = m - 1; t1 = m - 1; }
                else if (v == best_v) { t1 = m - 1; }
            }

            // 2) refine inside the local block around t0 (simple & crucial)
            // STEP=4 => check +/-3 to not miss a peak between sampled points.
            size_t lo_blk = (t0 >= (STEP - 1) ? t0 - (STEP - 1) : 0);
            size_t hi_blk = std::min(m - 1, t1 + (STEP - 1));
            size_t t_peak = t0;
            for (size_t t = lo_blk; t <= hi_blk; ++t) {
                size_t v = getv(t);
                if (v > best_v) { best_v = v; t_peak = t; }
            }

            return {t_peak, best_v};
        };

        if (i < n - 1 && keys[i] + 1 != keys[i + 1]) {
            std::optional<uint64_t> last_right;
            last_right = skipset_stepper.last_right(keys[i], poison_budget);
            if (last_right.has_value()) {
                uint64_t last_right_value = last_right.value();
                size_t k_index_of_last_right = std::lower_bound(keys.begin(), keys.end(), last_right_value) - keys.begin();
                assert(k_index_of_last_right < n);

                // Evaluate "max reachable segment end" for a fixed c (RIGHT case)
                auto eval_right = [&](size_t c) -> size_t {
                    double intercept_at_key0 = intercept_at_key0_candidates[c];
                    const SparseTableMaxLD& st_min_slope_right_of_poisons = st_min_slope_right_of_poisons_for_each_intercept[c];
                    const SparseTableMinLD& st_max_slope_right_of_poisons = st_max_slope_right_of_poisons_for_each_intercept[c];

                    SlopeRange slope_range_left = {
                        pref_max_min_slope[idx(c, i)],
                        pref_min_max_slope[idx(c, i)]
                    };

                    SlopeRange slope_range_last_right_poison = {
                        (static_cast<double>(k_index_of_last_right) + poison_budget_ld - 1.0 - epsilon_ld - intercept_at_key0) / static_cast<double>(last_right_value - keys[0]),
                        (static_cast<double>(k_index_of_last_right) + poison_budget_ld - 1.0 + epsilon_ld - intercept_at_key0) / static_cast<double>(last_right_value - keys[0])
                    };

                    SlopeRange slope_range_after_last_right_poison = {
                        suff_max_min_slope_right[idx(c, k_index_of_last_right)],
                        suff_min_max_slope_right[idx(c, k_index_of_last_right)]
                    };

                    SlopeRange slope_range_combined = intersect(intersect(slope_range_left, slope_range_last_right_poison), slope_range_after_last_right_poison);
                    if (slope_range_combined.is_feasible()) {
                        return n - 1;
                    }

                    SlopeRange slope_range_left_and_poisons = intersect(slope_range_left, slope_range_last_right_poison);
                    if (!slope_range_left_and_poisons.is_feasible()) {
                        if (k_index_of_last_right == 0) {
                            return 0;
                        }
                        return k_index_of_last_right - 1;
                    }

                    // Binary search for the furthest feasible index
                    size_t feasible_idx;
                    if (k_index_of_last_right == 0) {
                        feasible_idx = 0;
                    } else {
                        feasible_idx = k_index_of_last_right - 1;
                    }
                    size_t infeasible_idx = n - 1;
                    while (feasible_idx + 1 < infeasible_idx) {
                        size_t mid = (feasible_idx + infeasible_idx) / 2;
                        SlopeRange slope_range_right_of_poisons_to_mid = {
                            st_min_slope_right_of_poisons.range_max(k_index_of_last_right, mid),
                            st_max_slope_right_of_poisons.range_min(k_index_of_last_right, mid)
                        };
                        SlopeRange slope_range_combined_to_mid = intersect(slope_range_left_and_poisons, slope_range_right_of_poisons_to_mid);
                        if (slope_range_combined_to_mid.is_feasible()) {
                            feasible_idx = mid;
                            if (feasible_idx >= optimal_segment_end_index_in_original_keys) break;
                        } else {
                            infeasible_idx = mid;
                        }
                    }
                    return feasible_idx;
                };

                // Find c that maximizes eval_right(c) using discrete golden search
                size_t max_segment_end_index_in_original_keys = 0;
                if (!c_indices.empty()) {
                    auto [_, best_v] = argmax_unimodal_on_c_indices(eval_right);
                    max_segment_end_index_in_original_keys = best_v;
                }

                // size_t max_segment_end_index_in_original_keys_2 = 0;
                // for (size_t c = 0; c < C; ++c) {
                //     size_t v = eval_right(c);
                //     if (v > max_segment_end_index_in_original_keys_2) {
                //         max_segment_end_index_in_original_keys_2 = v;
                //     }
                // }
                // if (max_segment_end_index_in_original_keys_2 != max_segment_end_index_in_original_keys) {
                //     std::cout << "max_segment_end_index_in_original_keys: " << max_segment_end_index_in_original_keys << std::endl;
                //     std::cout << "max_segment_end_index_in_original_keys_2: " << max_segment_end_index_in_original_keys_2 << std::endl;
                //     throw std::runtime_error("max_segment_end_index_in_original_keys != max_segment_end_index_in_original_keys_2");
                // }

                if (max_segment_end_index_in_original_keys < best_seg) {
                    best_seg = max_segment_end_index_in_original_keys;
                    best_is_right = true;
                    best_valid = true;
                }
            }
        }

        if (i > 0 && keys[i] - 1 != keys[i - 1]) { // only if not allow duplicates
            std::optional<uint64_t> last_left = skipset_stepper.last_left(keys[i], poison_budget);
            if (last_left.has_value()) {
                uint64_t last_left_value = last_left.value();
                size_t k_index_of_last_left = std::lower_bound(keys.begin(), keys.end(), last_left_value) - keys.begin();
                assert(k_index_of_last_left >= 1);

                // Evaluate "max reachable segment end" for a fixed c (LEFT case)
                auto eval_left = [&](size_t c) -> size_t {
                    double intercept_at_key0 = intercept_at_key0_candidates[c];
                    const SparseTableMaxLD& st_min_slope_right_of_poisons = st_min_slope_right_of_poisons_for_each_intercept[c];
                    const SparseTableMinLD& st_max_slope_right_of_poisons = st_max_slope_right_of_poisons_for_each_intercept[c];

                    SlopeRange slope_range_left = {
                        pref_max_min_slope[idx(c, k_index_of_last_left - 1)],
                        pref_min_max_slope[idx(c, k_index_of_last_left - 1)]
                    };

                    SlopeRange slope_range_last_left_poison = {
                        (static_cast<double>(k_index_of_last_left) - epsilon_ld - intercept_at_key0) / static_cast<double>(last_left_value - keys[0]),
                        (static_cast<double>(k_index_of_last_left) + epsilon_ld - intercept_at_key0) / static_cast<double>(last_left_value - keys[0])
                    };

                    SlopeRange slope_range_after_last_left_poison = {
                        suff_max_min_slope_right[idx(c, i)],
                        suff_min_max_slope_right[idx(c, i)]
                    };

                    SlopeRange slope_range_combined = intersect(intersect(slope_range_left, slope_range_last_left_poison), slope_range_after_last_left_poison);
                    if (slope_range_combined.is_feasible()) {
                        return n - 1;
                    }

                    SlopeRange slope_range_left_and_poisons = intersect(slope_range_left, slope_range_last_left_poison);
                    if (!slope_range_left_and_poisons.is_feasible()) {
                        return k_index_of_last_left - 1;
                    }

                    // Binary search for the furthest feasible index
                    size_t feasible_idx = i;
                    size_t infeasible_idx = n - 1;
                    while (feasible_idx + 1 < infeasible_idx) {
                        size_t mid = (feasible_idx + infeasible_idx) / 2;
                        SlopeRange slope_range_after_last_left_poison_to_mid = {
                            st_min_slope_right_of_poisons.range_max(i, mid),
                            st_max_slope_right_of_poisons.range_min(i, mid)
                        };
                        SlopeRange slope_range_combined_to_mid =
                            intersect(slope_range_left_and_poisons, slope_range_after_last_left_poison_to_mid);
                        if (slope_range_combined_to_mid.is_feasible()) {
                            feasible_idx = mid;
                            if (feasible_idx >= optimal_segment_end_index_in_original_keys) break;
                        } else {
                            infeasible_idx = mid;
                        }
                    }
                    return feasible_idx;
                };

                size_t max_segment_end_index_in_original_keys = 0;
                if (!c_indices.empty()) {
                    auto [_, best_v] = argmax_unimodal_on_c_indices(eval_left);
                    max_segment_end_index_in_original_keys = best_v;
                }

                // size_t max_segment_end_index_in_original_keys_2 = 0;
                // for (size_t c = 0; c < C; ++c) {
                //     size_t v = eval_left(c);
                //     if (v > max_segment_end_index_in_original_keys_2) {
                //         max_segment_end_index_in_original_keys_2 = v;
                //     }
                // }
                // if (max_segment_end_index_in_original_keys_2 != max_segment_end_index_in_original_keys) {
                //     std::cout << "max_segment_end_index_in_original_keys: " << max_segment_end_index_in_original_keys << std::endl;
                //     std::cout << "max_segment_end_index_in_original_keys_2: " << max_segment_end_index_in_original_keys_2 << std::endl;
                //     throw std::runtime_error("max_segment_end_index_in_original_keys != max_segment_end_index_in_original_keys_2");
                // }

                if (max_segment_end_index_in_original_keys < best_seg) {
                    best_seg = max_segment_end_index_in_original_keys;
                    best_is_right = false;
                    best_valid = true;
                }
            }
        }

        return SwingPerICandidate{best_seg, i, best_is_right, best_valid};
    };

    std::vector<size_t> c_indices(C);
    std::iota(c_indices.begin(), c_indices.end(), 0);

    const size_t n_small = std::min(n, (size_t)(10));
    std::vector<size_t> i_indices_small(n_small);
    for (size_t t = 0; t < n_small; ++t) {
        i_indices_small[t] = t * (n - 1) / (n_small - 1);
    }
    std::vector<SwingPerICandidate> best_per_i_small(n_small, SwingPerICandidate{n, 0, false, false});
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long t = 0; t < (long long)n_small; ++t) {
        const size_t i = i_indices_small[(size_t)t];
        best_per_i_small[(size_t)t] = get_best_per_i(i, c_indices);
    }

    for (const auto& cand : best_per_i_small) {
        if (cand.valid) {
            _update_optimal_poisons(
                optimal_i, optimal_is_right_of_key_i, optimal_segment_end_index_in_original_keys,
                cand.i, cand.is_right_of_key_i, cand.segment_end_index_in_original_keys
            );
        }
    }

    std::vector<SwingPerICandidate> best_per_i(optimal_segment_end_index_in_original_keys + 1, SwingPerICandidate{n, 0, false, false});
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
    for (long long t = 0; t < (long long)(optimal_segment_end_index_in_original_keys + 1); ++t) {
        const size_t i = (size_t)t;
        best_per_i[(size_t)t] = get_best_per_i(i, c_indices);
    }

    for (const auto& cand : best_per_i) {
        if (cand.valid) {
            _update_optimal_poisons(
                optimal_i, optimal_is_right_of_key_i, optimal_segment_end_index_in_original_keys,
                cand.i, cand.is_right_of_key_i, cand.segment_end_index_in_original_keys
            );
        }
    }

    if (optimal_i == n) {
        return std::vector<uint64_t>{};
    }
    std::vector<uint64_t> optimal_poisons;
    std::unordered_set<uint64_t> key_set(keys.begin(), keys.end());
    if (optimal_is_right_of_key_i) {
        uint64_t value = keys[optimal_i] + 1;
        while (optimal_poisons.size() < poison_budget) {
            if (key_set.find(value) == key_set.end()) {
                optimal_poisons.push_back(value);
            }
            ++value;
        }
    } else {
        uint64_t value = keys[optimal_i] - 1;
        while (optimal_poisons.size() < poison_budget) {
            if (key_set.find(value) == key_set.end()) {
                optimal_poisons.push_back(value);
            }
            --value;
        }
    }

    return optimal_poisons;
}




inline std::vector<uint64_t> compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl_with_sparse_table_allow_duplicates(
    const std::vector<uint64_t>& reduced_keys,
    const std::vector<uint64_t>& reduced_ranks,
    size_t epsilon,
    size_t poison_budget,
    std::vector<double> intercept_at_key0_candidates,
    const std::vector<double>& pref_max_min_slope,
    const std::vector<double>& pref_min_max_slope,
    const std::vector<double>& suff_max_min_slope_right,
    const std::vector<double>& suff_min_max_slope_right,
    size_t n,
    const std::vector<SparseTableMaxLD>& st_min_slope_right_of_poisons_for_each_intercept,
    const std::vector<SparseTableMinLD>& st_max_slope_right_of_poisons_for_each_intercept
) {
    auto idx = [n](size_t c, size_t i) -> size_t { return c * n + i; };
    const size_t C = intercept_at_key0_candidates.size();
    const double epsilon_ld = static_cast<double>(epsilon);
    const double poison_budget_ld = static_cast<double>(poison_budget);

    for (const auto& intercept_at_key0 : intercept_at_key0_candidates) {
        if (intercept_at_key0 < -epsilon_ld || epsilon_ld < intercept_at_key0) {
            throw std::invalid_argument("intercept_at_key0 is out of range");
        }
    }

    size_t optimal_i = n;
    bool optimal_is_right_of_key_i = false;
    size_t optimal_segment_end_index_in_original_keys = n - 1;

    auto get_best_per_i = [&](size_t i, const std::vector<size_t>& c_indices) {
        size_t best_seg = n;
        bool best_is_right = false;
        bool best_valid = false;

        // ------------------------------------------------------------
        // Returns argmax index in [0, m-1] and its value.
        // Assumption: f is unimodal on the order of c_indices.
        // ------------------------------------------------------------
        auto argmax_unimodal_on_c_indices = [&](auto&& f) -> std::pair<size_t, size_t> {
            const size_t m = c_indices.size();
            if (m == 0) return {0, 0};
            std::vector<char> seen(m, 0);
            std::vector<size_t> val(m, 0);
            auto getv = [&](size_t t) -> size_t {
                if (!seen[t]) {
                    seen[t] = 1;
                    val[t] = f(c_indices[t]);
                }
                return val[t];
            };
            if (m == 1) return {0, getv(0)};
            const size_t STEP = std::max(static_cast<size_t>(2), static_cast<size_t>(std::sqrt(m / 2)));
            // 1) coarse scan: every 4th
            size_t t0 = 0;
            size_t t1 = 0;
            size_t best_v = getv(0);
            for (size_t t = 0; t < m; t += STEP) {
                size_t v = getv(t);
                if (v > best_v) { best_v = v; t0 = t; t1 = t; }
                else if (v == best_v) { t1 = t; }
            }
            // also check last (often missed by step)
            {
                size_t v = getv(m - 1);
                if (v > best_v) { best_v = v; t0 = m - 1; t1 = m - 1; }
                else if (v == best_v) { t1 = m - 1; }
            }

            // 2) refine inside the local block around t0 (simple & crucial)
            // STEP=4 => check +/-3 to not miss a peak between sampled points.
            size_t lo_blk = (t0 >= (STEP - 1) ? t0 - (STEP - 1) : 0);
            size_t hi_blk = std::min(m - 1, t1 + (STEP - 1));
            size_t t_peak = t0;
            for (size_t t = lo_blk; t <= hi_blk; ++t) {
                size_t v = getv(t);
                if (v > best_v) { best_v = v; t_peak = t; }
            }

            return {t_peak, best_v};
        };

        uint64_t last_right_value = reduced_keys[i];
        const double rank_i_ld = (double)reduced_ranks[i];

        // Evaluate "max reachable segment end" for a fixed c (RIGHT case)
        auto eval_right = [&](size_t c) -> size_t {
            double intercept_at_key0 = intercept_at_key0_candidates[c];
            const SparseTableMaxLD& st_min_slope_right_of_poisons = st_min_slope_right_of_poisons_for_each_intercept[c];
            const SparseTableMinLD& st_max_slope_right_of_poisons = st_max_slope_right_of_poisons_for_each_intercept[c];

            SlopeRange slope_range_left = {
                pref_max_min_slope[idx(c, i)],
                pref_min_max_slope[idx(c, i)]
            };

            SlopeRange slope_range_last_right_poison = {
                0.0,
                std::numeric_limits<double>::infinity()
            };

            if (last_right_value != reduced_keys[0]) {
                slope_range_last_right_poison = {
                    (rank_i_ld + poison_budget_ld - 1.0 - epsilon_ld - intercept_at_key0) / static_cast<double>(last_right_value - reduced_keys[0]),
                    (rank_i_ld + poison_budget_ld - 1.0 + epsilon_ld - intercept_at_key0) / static_cast<double>(last_right_value - reduced_keys[0])
                };
            }

            SlopeRange slope_range_after_last_right_poison = {
                suff_max_min_slope_right[idx(c, i)],
                suff_min_max_slope_right[idx(c, i)]
            };

            SlopeRange slope_range_combined = intersect(intersect(slope_range_left, slope_range_last_right_poison), slope_range_after_last_right_poison);
            if (slope_range_combined.is_feasible()) {
                return n - 1;
            }

            SlopeRange slope_range_left_and_poisons = intersect(slope_range_left, slope_range_last_right_poison);
            if (!slope_range_left_and_poisons.is_feasible()) {
                if (i == 0) {
                    return 0;
                }
                return i - 1;
            }

            // Binary search for the furthest feasible index
            size_t feasible_idx;
            if (i == 0) {
                feasible_idx = 0;
            } else {
                feasible_idx = i - 1;
            }
            size_t infeasible_idx = n - 1;
            while (feasible_idx + 1 < infeasible_idx) {
                size_t mid = (feasible_idx + infeasible_idx) / 2;
                SlopeRange slope_range_right_of_poisons_to_mid = {
                    st_min_slope_right_of_poisons.range_max(i, mid),
                    st_max_slope_right_of_poisons.range_min(i, mid)
                };
                SlopeRange slope_range_combined_to_mid = intersect(slope_range_left_and_poisons, slope_range_right_of_poisons_to_mid);
                if (slope_range_combined_to_mid.is_feasible()) {
                    feasible_idx = mid;
                    if (feasible_idx >= optimal_segment_end_index_in_original_keys) break;
                } else {
                    infeasible_idx = mid;
                }
            }
            return feasible_idx;
        };

        // Find c that maximizes eval_right(c) using discrete golden search
        size_t max_segment_end_index_in_original_keys = 0;
        if (!c_indices.empty()) {
            auto [_, best_v] = argmax_unimodal_on_c_indices(eval_right);
            max_segment_end_index_in_original_keys = best_v;
        }

        if (max_segment_end_index_in_original_keys < best_seg) {
            best_seg = max_segment_end_index_in_original_keys;
            best_is_right = true;
            best_valid = true;
        }

        return SwingPerICandidate{best_seg, i, best_is_right, best_valid};
    };

    std::vector<size_t> c_indices(C);
    std::iota(c_indices.begin(), c_indices.end(), 0);

    const size_t n_small = std::min(n, (size_t)(10));
    std::vector<size_t> i_indices_small(n_small);
    for (size_t t = 0; t < n_small; ++t) {
        i_indices_small[t] = t * (n - 1) / (n_small - 1);
    }
    std::vector<SwingPerICandidate> best_per_i_small(n_small, SwingPerICandidate{n, 0, false, false});
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long t = 0; t < (long long)n_small; ++t) {
        const size_t i = i_indices_small[(size_t)t];
        best_per_i_small[(size_t)t] = get_best_per_i(i, c_indices);
    }

    for (const auto& cand : best_per_i_small) {
        if (cand.valid) {
            _update_optimal_poisons(
                optimal_i, optimal_is_right_of_key_i, optimal_segment_end_index_in_original_keys,
                cand.i, cand.is_right_of_key_i, cand.segment_end_index_in_original_keys
            );
        }
    }

    std::vector<SwingPerICandidate> best_per_i(optimal_segment_end_index_in_original_keys + 1, SwingPerICandidate{n, 0, false, false});
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
    for (long long t = 0; t < (long long)(optimal_segment_end_index_in_original_keys + 1); ++t) {
        const size_t i = (size_t)t;
        best_per_i[(size_t)t] = get_best_per_i(i, c_indices);
    }

    for (const auto& cand : best_per_i) {
        if (cand.valid) {
            _update_optimal_poisons(
                optimal_i, optimal_is_right_of_key_i, optimal_segment_end_index_in_original_keys,
                cand.i, cand.is_right_of_key_i, cand.segment_end_index_in_original_keys
            );
        }
    }

    if (optimal_i == n) {
        return std::vector<uint64_t>{};
    }
    std::vector<uint64_t> optimal_poisons;
    optimal_poisons.resize(poison_budget);
    std::fill(optimal_poisons.begin(), optimal_poisons.end(), reduced_keys[optimal_i]);

    return optimal_poisons;
}



/**
 * @brief Finds poisons that minimize segment length for a given poison budget using a swing.
 *
 * @param keys Sorted input keys
 * @param epsilon Epsilon parameter for PGM-index
 * @param poison_budget Number of poisons to inject
 * @param intercept_at_key0_candidates Candidates for intercept at key[0]
 * @param allow_duplicates Whether duplicate keys in keys are allowed
 * @return Optimal poisons (empty if none)
 */
inline std::vector<uint64_t> compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl(
    const std::vector<uint64_t>& keys,
    size_t epsilon,
    size_t poison_budget,
    std::vector<double> intercept_at_key0_candidates,
    bool allow_duplicates = true
) {
    if (keys.empty()) {
        throw std::invalid_argument("keys is empty");
    }
    if (keys.size() < 2) {
        return std::vector<uint64_t>{};
    }
    if (poison_budget == 0) {
        return std::vector<uint64_t>{};
    }

    std::vector<uint64_t> reduced_keys, reduced_ranks;
    {
        size_t n = keys.size();
        reduced_keys.push_back(keys[0]);
        reduced_ranks.push_back(0);
        for (size_t i = 1; i < n - 1; ++i) {
            if (keys[i] == keys[i - 1]) {
                if (keys[i] + 1 < keys[i + 1]) {
                    reduced_keys.push_back(keys[i] + 1);
                    reduced_ranks.push_back(i);
                }
            } else {
                reduced_keys.push_back(keys[i]);
                reduced_ranks.push_back(i);
            }
        }
        if (n >= 2 && keys[n - 1] != keys[n - 2]) {
            reduced_keys.push_back(keys[n - 1]);
            reduced_ranks.push_back(n - 1);
        }
    }

    {
        MinimaxFitResult minimax_fit_result = minimax_maxabs_regression_rank(reduced_keys, reduced_ranks);
        double max_error = minimax_fit_result.max_error;
        if (max_error + 0.5 * poison_budget < epsilon) {
            // Even after adding poison_budget poisons, the max error is still less than epsilon, so no poisons are needed.
            // std::cout << "OK" << std::endl;
            return std::vector<uint64_t>{};
        }
    }

    size_t n = reduced_keys.size();
    size_t C = intercept_at_key0_candidates.size();
    double epsilon_ld = static_cast<double>(epsilon);
    double poison_budget_ld = static_cast<double>(poison_budget);

    for (const auto& intercept_at_key0 : intercept_at_key0_candidates) {
        if (intercept_at_key0 < -epsilon_ld || epsilon_ld < intercept_at_key0) {
            throw std::invalid_argument("intercept_at_key0 is out of range");
        }
    }

    std::vector<double> inv_keys_i_minus_keys_0(n);
    for (size_t i = 0; i < n; ++i) {
        inv_keys_i_minus_keys_0[i] = 1.0 / static_cast<double>(reduced_keys[i] - reduced_keys[0]);
    }

    auto idx = [n](size_t c, size_t i) -> size_t { return c * n + i; };

    std::vector<double> pref_max_min_slope(C * n);
    std::vector<double> pref_min_max_slope(C * n);
    std::vector<double> suff_max_min_slope_right(C * n);
    std::vector<double> suff_min_max_slope_right(C * n);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long c = 0; c < (long long)C; ++c) {
        const size_t c_idx = (size_t)c;
        const double b0 = intercept_at_key0_candidates[c];

        double* p_pref_maxmin = &pref_max_min_slope[c*n];
        double* p_pref_minmax = &pref_min_max_slope[c*n];
        double* p_suff_maxminR = &suff_max_min_slope_right[c*n];
        double* p_suff_minmaxR = &suff_min_max_slope_right[c*n];

        // ---- pref (left) ----
        // i=0
        double run_max_min = std::numeric_limits<double>::lowest();
        double run_min_max = std::numeric_limits<double>::max();
        p_pref_maxmin[0] = run_max_min;
        p_pref_minmax[0] = run_min_max;

        double minus_epsilon_ld_minus_b0 = - epsilon_ld - b0;
        double plus_epsilon_ld_minus_b0 = epsilon_ld - b0;

        for (size_t i = 1; i < n; ++i) {
            const double rank_i_ld = (double)reduced_ranks[i];
            const double mn = (rank_i_ld + minus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[i];
            const double mx = (rank_i_ld + plus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[i];
            run_max_min = std::max(run_max_min, mn);
            run_min_max = std::min(run_min_max, mx);
            p_pref_maxmin[i] = run_max_min;
            p_pref_minmax[i] = run_min_max;
        }

        // ---- suff (right-of-poisons) ----
        {
            double poison_budget_ld_minus_epsilon_ld_minus_b0 = poison_budget_ld - epsilon_ld - b0;
            double poison_budget_ld_plus_epsilon_ld_minus_b0 = poison_budget_ld + epsilon_ld - b0;

            const size_t i = n - 1;
            const double rank_i_ld = (double)reduced_ranks[i];
            double run_max_minR = (rank_i_ld + poison_budget_ld_minus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[i];
            double run_min_maxR = (rank_i_ld + poison_budget_ld_plus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[i];
            p_suff_maxminR[i] = run_max_minR;
            p_suff_minmaxR[i] = run_min_maxR;

            for (size_t k = n - 1; k > 0; --k) {
                const size_t j = k - 1;
                const double rank_j_ld = (double)reduced_ranks[j];
                const double mnR = (rank_j_ld + poison_budget_ld_minus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[j];
                const double mxR = (rank_j_ld + poison_budget_ld_plus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[j];
                run_max_minR = std::max(run_max_minR, mnR);
                run_min_maxR = std::min(run_min_maxR, mxR);
                p_suff_maxminR[j] = run_max_minR;
                p_suff_minmaxR[j] = run_min_maxR;
            }
        }
    }

    // Check if dup-allowed attack can shorten the segment
    {
        bool can_shorten = false;
        for (size_t i = 0; i < n; ++i) {
            bool can_cover_all_keys = false;
            for (size_t c = 0; c < C; ++c) {
                SlopeRange slope_range_left = {
                    pref_max_min_slope[idx(c, i)],
                    pref_min_max_slope[idx(c, i)]
                };
                SlopeRange slope_range_right = {
                    suff_max_min_slope_right[idx(c, i)],
                    suff_min_max_slope_right[idx(c, i)]
                };
                SlopeRange slope_range_combined = intersect(slope_range_left, slope_range_right);
                if (slope_range_combined.is_feasible()) {
                    can_cover_all_keys = true;
                    break;
                }
            }
            if (!can_cover_all_keys) {
                can_shorten = true;
                break;
            }
        }
        if (!can_shorten) {
            return std::vector<uint64_t>{};
        }
    }

    std::vector<SparseTableMaxLD> st_min_slope_right_of_poisons_for_each_intercept(C);
    std::vector<SparseTableMinLD> st_max_slope_right_of_poisons_for_each_intercept(C);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long c = 0; c < (long long)C; ++c) {
        const size_t c_idx = (size_t)c;
        const double b0 = intercept_at_key0_candidates[c];

        double poison_budget_ld_minus_epsilon_ld_minus_b0 = poison_budget_ld - epsilon_ld - b0;
        double poison_budget_ld_plus_epsilon_ld_minus_b0 = poison_budget_ld + epsilon_ld - b0;

        std::vector<double> tmp;
        tmp.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const double rank_i_ld = (double)reduced_ranks[i];
            tmp[i] = (rank_i_ld + poison_budget_ld_minus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[i];
        }
        st_min_slope_right_of_poisons_for_each_intercept[c_idx] = SparseTableMaxLD(tmp);

        for (size_t i = 0; i < n; ++i) {
            const double rank_i_ld = (double)reduced_ranks[i];
            tmp[i] = (rank_i_ld + poison_budget_ld_plus_epsilon_ld_minus_b0) * inv_keys_i_minus_keys_0[i];
        }
        st_max_slope_right_of_poisons_for_each_intercept[c_idx] = SparseTableMinLD(tmp);
    }

    if (allow_duplicates) {
        return compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl_with_sparse_table_allow_duplicates(
            reduced_keys, reduced_ranks,
            epsilon, poison_budget,
            intercept_at_key0_candidates,
            pref_max_min_slope,
            pref_min_max_slope,
            suff_max_min_slope_right,
            suff_min_max_slope_right,
            n,
            st_min_slope_right_of_poisons_for_each_intercept,
            st_max_slope_right_of_poisons_for_each_intercept
        );
    } else {
        SkipSetStepper skipset_stepper(reduced_keys);
        return compute_consecutive_multiple_poisons_minimize_segment_length_swing_impl_with_sparse_table(
            reduced_keys, epsilon, poison_budget,
            intercept_at_key0_candidates,
            pref_max_min_slope,
            pref_min_max_slope,
            suff_max_min_slope_right,
            suff_min_max_slope_right,
            n,
            st_min_slope_right_of_poisons_for_each_intercept,
            st_max_slope_right_of_poisons_for_each_intercept,
            skipset_stepper
        );
    }
}

} // namespace minimize_segment_length_swing_common

#endif // MINIMIZE_SEGMENT_LENGTH_SWING_COMMON_HPP
