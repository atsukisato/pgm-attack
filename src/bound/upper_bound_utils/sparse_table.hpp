#ifndef UPPER_BOUND_UTILS_SPARSE_TABLE_HPP
#define UPPER_BOUND_UTILS_SPARSE_TABLE_HPP

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

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
        if (l > r) return std::numeric_limits<ld>::lowest();
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

#endif  // UPPER_BOUND_UTILS_SPARSE_TABLE_HPP
