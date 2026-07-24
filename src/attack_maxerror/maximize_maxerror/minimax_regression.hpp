#pragma once

#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>

struct MinimaxFitResult {
    double slope;     // a
    double intercept; // b
    double max_error; // max_i |y_i - (a*x_i + b)|  (half of the residual range)
};

// Minimax (L-infinity / Chebyshev) regression for points:
//   x_i = keys[i], y_i = i   (0-based rank)
// Returns slope a, intercept b, and max_error (half of max residual - min residual).
MinimaxFitResult minimax_maxabs_regression_rank(const std::vector<uint64_t>& keys) {
    if (!std::is_sorted(keys.begin(), keys.end())) {
        throw std::invalid_argument("keys is not sorted");
    }

    const size_t n = keys.size();
    if (n == 0) {
        return {0.0, 0.0, 0.0};
    }
    if (n == 1) {
        return {0.0, 0.0, 0.0};
    }
    if (n == 2) {
        const uint64_t x0 = keys[0], x1 = keys[1];
        if (x1 == x0) {
            return {0.0, 0.5, 0.5};  // degenerate: same x, loss cannot be 0
        }
        double slope = 1.0 / static_cast<double>(x1 - x0);
        double intercept = -slope * static_cast<double>(x0);
        return {slope, intercept, 0.0};  // line through (x0,0) and (x1,1), loss = 0
    }

    const uint64_t x0 = keys[0];  // shift by key[0] for numerical stability
    using ld = long double;
    const ld INF = std::numeric_limits<ld>::infinity();

    auto eval_at = [&](ld a, ld& g_out, ld& h_out,
                       uint64_t& x_at_max, uint64_t& x_at_min) {
        // t_i = y_i - a*x'_i  with x'_i = keys[i] - x0
        ld g = -INF; // max t
        ld h = +INF; // min t
        size_t i_max = 0, i_min = 0;

        for (size_t i = 0; i < n; ++i) {
            const ld x = (ld)(keys[i] - x0);
            const ld y = (ld)i;               // 0-based rank
            const ld t = y - a * x;

            if (t > g) { g = t; i_max = i; }
            if (t < h) { h = t; i_min = i; }
        }
        g_out = g;
        h_out = h;
        x_at_max = keys[i_max];
        x_at_min = keys[i_min];
    };

    // We minimize f(a) = (max t_i) - (min t_i). This is convex.
    // Use subgradient sign based on one argmax/argmin:
    //   f'(a) contains (x_min - x_max). If negative => f decreasing => move right (increase a).
    // If positive => f increasing => move left (decrease a).
    auto grad_sign = [&](ld a) -> int {
        ld g, h;
        uint64_t x_max, x_min;
        eval_at(a, g, h, x_max, x_min);
        if (x_min > x_max) return +1; // positive
        if (x_min < x_max) return -1; // negative
        return 0;
    };

    // Bracket the minimizer: at a=0, typically negative; at large a, positive.
    ld lo = 0.0L;
    ld hi = 1.0L;

    // Ensure hi is large enough so grad_sign(hi) >= 0 (or ==0).
    // Put a reasonable cap to avoid infinite loop in pathological numeric cases.
    for (int iter = 0; iter < 200; ++iter) {
        const int s = grad_sign(hi);
        if (s >= 0) break;
        hi *= 2.0L;
        if (!std::isfinite((double)hi)) { // fallback
            hi = 1e30L;
            break;
        }
    }

    // Bisection on subgradient sign
    for (int iter = 0; iter < 90; ++iter) {
        const ld mid = (lo + hi) * 0.5L;
        const int s = grad_sign(mid);
        if (s == 0) { lo = hi = mid; break; }
        if (s < 0) lo = mid;  // negative: move right
        else       hi = mid;  // positive: move left
    }

    const ld a_star = (lo + hi) * 0.5L;

    // Compute optimal b for this a_star and the resulting 2*max_error = g-h
    ld g, h;
    uint64_t x_max, x_min;
    eval_at(a_star, g, h, x_max, x_min);

    const ld b_star = (g + h) * 0.5L;
    const ld two_err = (g - h);
    const ld max_err = two_err * 0.5L;
    const ld b_original = b_star - a_star * (ld)x0;  // intercept in original key space

    // Return as double (max_error = two_max_error / 2)
    return {
        (double)a_star,
        (double)b_original,
        (double)max_err
    };
}



// Minimax (L-infinity / Chebyshev) regression for points:
//   (x_i, y_i) = (keys[i], ranks[i])
// Returns slope a, intercept b, and max_error.
inline MinimaxFitResult minimax_maxabs_regression_rank(
    const std::vector<uint64_t>& keys,
    const std::vector<uint64_t>& ranks) {

    if (keys.size() != ranks.size()) {
        throw std::invalid_argument("keys.size() != ranks.size()");
    }
    if (!std::is_sorted(keys.begin(), keys.end())) {
        throw std::invalid_argument("keys is not sorted");
    }

    const size_t n = keys.size();
    if (n == 0) {
        return {0.0, 0.0, 0.0};
    }
    if (n == 1) {
        return {0.0, static_cast<double>(ranks[0]), 0.0};
    }
    if (n == 2) {
        const uint64_t x0 = keys[0], x1 = keys[1];
        const uint64_t y0 = ranks[0], y1 = ranks[1];

        if (x1 == x0) {
            // Same x: cannot fit both exactly unless y0 == y1.
            const double intercept = 0.5 * (static_cast<double>(y0) + static_cast<double>(y1));
            const double max_error =
                0.5 * std::abs(static_cast<double>(y1) - static_cast<double>(y0));
            return {0.0, intercept, max_error};
        }

        const double slope =
            (static_cast<double>(y1) - static_cast<double>(y0)) /
            static_cast<double>(x1 - x0);
        const double intercept = static_cast<double>(y0) - slope * static_cast<double>(x0);
        return {slope, intercept, 0.0};
    }

    const uint64_t x0 = keys[0];  // shift for numerical stability
    using ld = long double;
    const ld INF = std::numeric_limits<ld>::infinity();

    auto eval_at = [&](ld a, ld& g_out, ld& h_out,
                       uint64_t& x_at_max, uint64_t& x_at_min) {
        // t_i = y_i - a * x'_i, where x'_i = keys[i] - x0
        // Objective: minimize max_i t_i - min_i t_i
        ld g = -INF; // max t_i
        ld h = +INF; // min t_i
        size_t i_max = 0, i_min = 0;

        for (size_t i = 0; i < n; ++i) {
            const ld x = static_cast<ld>(keys[i] - x0);
            const ld y = static_cast<ld>(ranks[i]);
            const ld t = y - a * x;

            if (t > g) { g = t; i_max = i; }
            if (t < h) { h = t; i_min = i; }
        }

        g_out = g;
        h_out = h;
        x_at_max = keys[i_max];
        x_at_min = keys[i_min];
    };

    // For f(a) = max_i(y_i - a x_i') - min_i(y_i - a x_i'),
    // a subgradient is in the interval [x_min - x_max].
    // Using one argmax/argmin pair:
    //   if x_min > x_max => subgradient positive  => move left
    //   if x_min < x_max => subgradient negative  => move right
    auto grad_sign = [&](ld a) -> int {
        ld g, h;
        uint64_t x_max, x_min;
        eval_at(a, g, h, x_max, x_min);
        if (x_min > x_max) return +1;
        if (x_min < x_max) return -1;
        return 0;
    };

    // Find a bracket [lo, hi] containing an optimum.
    // Unlike the rank=i case, slope can be negative in general,
    // so we expand symmetrically if needed.
    ld lo = -1.0L;
    ld hi = +1.0L;

    for (int iter = 0; iter < 200; ++iter) {
        const int s_lo = grad_sign(lo);
        const int s_hi = grad_sign(hi);

        if (s_lo <= 0 && s_hi >= 0) break;

        lo *= 2.0L;
        hi *= 2.0L;

        if (!std::isfinite(static_cast<double>(lo)) ||
            !std::isfinite(static_cast<double>(hi))) {
            lo = -1e30L;
            hi = +1e30L;
            break;
        }
    }

    // Bisection on subgradient sign
    for (int iter = 0; iter < 100; ++iter) {
        const ld mid = (lo + hi) * 0.5L;
        const int s = grad_sign(mid);

        if (s == 0) {
            lo = hi = mid;
            break;
        }
        if (s < 0) lo = mid;  // decreasing => move right
        else       hi = mid;  // increasing => move left
    }

    const ld a_star = (lo + hi) * 0.5L;

    ld g, h;
    uint64_t x_max, x_min;
    eval_at(a_star, g, h, x_max, x_min);

    // Optimal intercept for fixed slope a_star
    const ld b_shifted = (g + h) * 0.5L;
    const ld max_err = (g - h) * 0.5L;
    const ld b_original = b_shifted - a_star * static_cast<ld>(x0);

    return {
        static_cast<double>(a_star),
        static_cast<double>(b_original),
        static_cast<double>(max_err)
    };
}
