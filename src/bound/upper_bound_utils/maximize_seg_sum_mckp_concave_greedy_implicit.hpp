#pragma once

#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace upper_bound_utils {

/**
 * Structure to hold the upper bound information for each block.
 * Contains only nu_values and D(nu), and f(λ) is evaluated on demand.
 */
struct BlockUpperBoundImplicit {
    const std::vector<double>* nu_values;
    std::vector<double> D;
};

/**
 * Evaluate the upper bound of the number of segments f_i(λ) = min_k ( (λ - D[k]) / nu[k] ) for each block.
 */
inline double eval_seg_num_upper_bound(const BlockUpperBoundImplicit& blk, size_t lambda) {
    const auto& nus = *blk.nu_values;
    double best = std::numeric_limits<double>::infinity();
    const double lam = static_cast<double>(lambda);

    for (size_t k = 0; k < nus.size(); ++k) {
        const double nu = nus[k];
        const double D_val = blk.D[k];
        const double cand = (lam - D_val) / nu;
        if (cand < best) best = cand;
    }
    return best;
}

/**
 * Return the increment gain = f(cur_lambda + 1) - f(cur_lambda) when λ is increased by 1.
 */
inline double gain_next(const BlockUpperBoundImplicit& blk, size_t cur_lambda) {
    const double f0 = eval_seg_num_upper_bound(blk, cur_lambda);
    const double f1 = eval_seg_num_upper_bound(blk, cur_lambda + 1);
    return f1 - f0;
}

/** For the heap: the increment gain and the next λ when increasing λ by 1 for block i */
struct HeapItemImplicit {
    double gain;
    int idx;
    size_t next_lam;
};

/** For the heap: max-heap by gain */
struct HeapCmpImplicit {
    bool operator()(const HeapItemImplicit& a, const HeapItemImplicit& b) const {
        return a.gain < b.gain;
    }
};

/**
 * Select λ_i from each block, and maximize sum f_i(λ_i) under sum λ_i <= lambda_all (greedy for discrete concave MCKP).
 * f_i(λ) is evaluated on demand. Computation time O((B + L) log B).
 *
 * @param blocks upper bound information for each block
 * @param lambda_all upper bound of the total λ
 * @return the sum of the maximized number of segments
 */
inline double maximize_seg_sum_mckp_concave_greedy_implicit(
    const std::vector<BlockUpperBoundImplicit>& blocks,
    size_t lambda_all) {
    const int B = (int)blocks.size();
    if (B == 0) return 0.0;

    double total = 0.0;
    std::vector<size_t> alloc(B, 0);

    std::priority_queue<HeapItemImplicit, std::vector<HeapItemImplicit>, HeapCmpImplicit> pq;

    for (int i = 0; i < B; ++i) {
        const double f0 = eval_seg_num_upper_bound(blocks[i], 0);
        if (!std::isfinite(f0)) {
            throw std::runtime_error("non-finite f_i(0) encountered in implicit evaluation");
        }
        total += f0;

        if (lambda_all >= 1) {
            const double g = gain_next(blocks[i], 0);
            if (std::isfinite(g)) {
                pq.push(HeapItemImplicit{g, i, 1});
            }
        }
    }

    for (size_t used = 0; used < lambda_all && !pq.empty(); ++used) {
        auto top = pq.top();
        pq.pop();

        if (top.next_lam != alloc[top.idx] + 1) {
            --used;
            continue;
        }

        if (!(top.gain > 0.0)) break;

        total += top.gain;
        alloc[top.idx]++;

        const int i = top.idx;
        const size_t cur = alloc[i];
        if (cur < lambda_all) {
            const double g = gain_next(blocks[i], cur);
            if (std::isfinite(g)) {
                pq.push(HeapItemImplicit{g, i, cur + 1});
            }
        }
    }

    return total;
}

}  // namespace upper_bound_utils
