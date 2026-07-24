#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "sparse_table.hpp"

namespace upper_bound_utils {
namespace detail {

/**
 * C[i][j] = max(0, 2*epsilon - (max(A[i..j]) - min(A[i..j])))
 * j <= i is inf
 */
inline double get_C_ij(const SparseTableMinLD& st_min, const SparseTableMaxLD& st_max,
                       double epsilon, int i, int j) {
    if (j <= i) return std::numeric_limits<double>::infinity();
    const double min_val = st_min.range_min(i, j);
    const double max_val = st_max.range_max(i, j);
    const double max_error_times_2 = max_val - min_val;
    return std::max(0.0, 2.0 * epsilon - max_error_times_2);
}

}  // namespace detail

/**
 * Use Monge structure to compute D(nu) = min_j dist[j] in O(n log n) using simplified online DP.
 * A[j][i] = dist[i] + C_ij  (i < j)
 * dist[j] = min_{i < j} A[j][i] - nu
 *
 * Corresponds to the random access version of "Simplified LARSCH Algorithm" by noshi91.
 */
inline double compute_D_nu_o_nlogn(const std::vector<double>& A, double nu, double epsilon) {
    const int n = static_cast<int>(A.size());
    if (n <= 1) return 0.0;
    if (nu <= 0.0) return std::numeric_limits<double>::infinity();

    const double INF = std::numeric_limits<double>::infinity();

    SparseTableMinLD st_min(A);
    SparseTableMaxLD st_max(A);

    std::vector<double> dist(n, INF);
    dist[0] = 0.0;
    std::vector<double> best_before_sub(n, INF);
    std::vector<int> argmin(n, 0);
    argmin[0] = 0;

    double answer = 0.0;
    auto eval = [&](int i, int j) -> double {
        if (i < 0 || j < 0 || i >= n || j >= n) return INF;
        if (j <= i) return INF;
        if (dist[i] == INF) return INF;
        const double c = detail::get_C_ij(st_min, st_max, epsilon, i, j);
        if (c == INF) return INF;
        return dist[i] + c;
    };

    auto check = [&](int row, int col) {
        const double v = eval(col, row);
        if (v < best_before_sub[row]) {
            best_before_sub[row] = v;
            argmin[row] = col;
        }
    };

    auto solve = [&](auto&& self, int l, int r) -> void {
        if (r - l == 1) return;
        const int m = (l + r) / 2;
        for (int k = argmin[l]; k <= argmin[r]; ++k) check(m, k);
        dist[m] = (best_before_sub[m] == INF ? INF : best_before_sub[m] - nu);
        if (dist[m] < answer) answer = dist[m];
        self(self, l, m);
        for (int k = l + 1; k <= m; ++k) check(r, k);
        dist[r] = (best_before_sub[r] == INF ? INF : best_before_sub[r] - nu);
        if (dist[r] < answer) answer = dist[r];
        self(self, m, r);
    };

    const int N = n - 1;
    check(N, 0);
    solve(solve, 0, N);

    return answer;
}

}  // namespace upper_bound_utils