#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "libquant/matrix.hpp"

namespace libquant {

// Singular Value Decomposition A = U diag(s) V^T for an m x n matrix with
// m >= n (for m < n, transpose A, decompose, and swap U/V).
//
// Two-phase Golub-Kahan-Reinsch algorithm (Golub & Van Loan, Sec. 8.6):
//   1. Householder bidiagonalization reduces A to upper-bidiagonal form
//      B = U1^T A V1 (Algorithm 5.4.2), accumulating U1, V1.
//   2. The implicit-shift QR algorithm for bidiagonal matrices (the
//      "Golub-Kahan SVD step", Algorithm 8.6.1) drives B to diagonal form
//      via Givens rotations applied from both sides, accumulated into U, V.
// This is the classical Golub-Reinsch/LINPACK dsvdc formulation. Singular
// values are returned in descending order.
template <typename T = double>
class SVD {
public:
    explicit SVD(Matrix<T> Ain) {
        const std::size_t m = Ain.rows();
        const std::size_t n = Ain.cols();
        if (m < n) throw std::invalid_argument("SVD: requires rows >= cols");

        Matrix<T> A = std::move(Ain);
        const std::size_t nu = n;
        s_.assign(n, T{0});
        std::vector<T> e(n, T{0});
        std::vector<T> work(m, T{0});
        U_ = Matrix<T>(m, nu, T{0});
        V_ = Matrix<T>(n, n, T{0});

        const std::size_t nct = std::min(m - 1, n);
        const std::size_t nrt = (n >= 2) ? std::min(n - 2, m) : 0;
        const std::size_t iters = std::max(nct, nrt);

        // Phase 1: Householder bidiagonalization (Golub & Van Loan, Alg. 5.4.2).
        for (std::size_t k = 0; k < iters; ++k) {
            if (k < nct) {
                s_[k] = T{0};
                for (std::size_t i = k; i < m; ++i) s_[k] = std::hypot(s_[k], A(i, k));
                if (s_[k] != T{0}) {
                    if (A(k, k) < T{0}) s_[k] = -s_[k];
                    for (std::size_t i = k; i < m; ++i) A(i, k) /= s_[k];
                    A(k, k) += T{1};
                }
                s_[k] = -s_[k];
            }
            for (std::size_t j = k + 1; j < n; ++j) {
                if (k < nct && s_[k] != T{0}) {
                    T t = T{0};
                    for (std::size_t i = k; i < m; ++i) t += A(i, k) * A(i, j);
                    t = -t / A(k, k);
                    for (std::size_t i = k; i < m; ++i) A(i, j) += t * A(i, k);
                }
                e[j] = A(k, j);
            }
            if (k < nct)
                for (std::size_t i = k; i < m; ++i) U_(i, k) = A(i, k);

            if (k < nrt) {
                e[k] = T{0};
                for (std::size_t i = k + 1; i < n; ++i) e[k] = std::hypot(e[k], e[i]);
                if (e[k] != T{0}) {
                    if (e[k + 1] < T{0}) e[k] = -e[k];
                    for (std::size_t i = k + 1; i < n; ++i) e[i] /= e[k];
                    e[k + 1] += T{1};
                }
                e[k] = -e[k];
                if (k + 1 < m && e[k] != T{0}) {
                    for (std::size_t i = k + 1; i < m; ++i) work[i] = T{0};
                    for (std::size_t j = k + 1; j < n; ++j)
                        for (std::size_t i = k + 1; i < m; ++i) work[i] += e[j] * A(i, j);
                    for (std::size_t j = k + 1; j < n; ++j) {
                        T t = -e[j] / e[k + 1];
                        for (std::size_t i = k + 1; i < m; ++i) A(i, j) += t * work[i];
                    }
                }
                for (std::size_t i = k + 1; i < n; ++i) V_(i, k) = e[i];
            }
        }

        // Trailing dimensions: with m >= n throughout, nu = p = n.
        const std::size_t p_dim = n;
        if (nct < n) s_[nct] = A(nct, nct);
        if (nrt + 1 < p_dim) e[nrt] = A(nrt, p_dim - 1);
        e[p_dim - 1] = T{0};

        // Accumulate U.
        for (std::size_t j = nct; j < nu; ++j) {
            for (std::size_t i = 0; i < m; ++i) U_(i, j) = T{0};
            U_(j, j) = T{1};
        }
        for (std::size_t kk = nct; kk-- > 0;) {
            std::size_t k = kk;
            if (s_[k] != T{0}) {
                for (std::size_t j = k + 1; j < nu; ++j) {
                    T t = T{0};
                    for (std::size_t i = k; i < m; ++i) t += U_(i, k) * U_(i, j);
                    t = -t / U_(k, k);
                    for (std::size_t i = k; i < m; ++i) U_(i, j) += t * U_(i, k);
                }
                for (std::size_t i = k; i < m; ++i) U_(i, k) = -U_(i, k);
                U_(k, k) = T{1} + U_(k, k);
                for (std::size_t i = 0; i + 1 < k; ++i) U_(i, k) = T{0};
            } else {
                for (std::size_t i = 0; i < m; ++i) U_(i, k) = T{0};
                U_(k, k) = T{1};
            }
        }

        // Accumulate V.
        for (std::size_t kk = n; kk-- > 0;) {
            std::size_t k = kk;
            if (k < nrt && e[k] != T{0}) {
                for (std::size_t j = k + 1; j < nu; ++j) {
                    T t = T{0};
                    for (std::size_t i = k + 1; i < n; ++i) t += V_(i, k) * V_(i, j);
                    t = -t / V_(k + 1, k);
                    for (std::size_t i = k + 1; i < n; ++i) V_(i, j) += t * V_(i, k);
                }
            }
            for (std::size_t i = 0; i < n; ++i) V_(i, k) = T{0};
            V_(k, k) = T{1};
        }

        diagonalize_bidiagonal(e, m, n);
    }

    [[nodiscard]] const std::vector<T>& singular_values() const noexcept { return s_; }
    [[nodiscard]] const Matrix<T>& U() const noexcept { return U_; }
    [[nodiscard]] const Matrix<T>& V() const noexcept { return V_; }

private:
    // Phase 2: implicit-shift QR sweep on the bidiagonal (s_, e) -- the
    // classical Golub-Reinsch / LINPACK dsvdc bidiagonal SVD step (Golub &
    // Van Loan, Algorithm 8.6.1), applying Givens rotations to U_ and V_.
    // Uses signed indices throughout since the reference algorithm relies
    // on a "no split found" sentinel of k == -1.
    void diagonalize_bidiagonal(std::vector<T>& e, std::size_t m, std::size_t n) {
        using Idx = long long;
        const auto S = [](std::size_t x) { return static_cast<std::size_t>(x); };
        const Idx N = static_cast<Idx>(n);
        const Idx M = static_cast<Idx>(m);
        Idx p = N;
        const Idx pp = p - 1;  // fixed for the whole sweep: see note below.
        int iter = 0;
        const T eps = std::numeric_limits<T>::epsilon();
        const T tiny = std::numeric_limits<T>::min();

        // `pp` stays fixed at n-1 for the entire routine (not re-derived from
        // the shrinking active window p): case 4 below bubble-inserts each
        // newly converged singular value up past already-placed larger
        // entries using this fixed upper bound, which is what produces a
        // fully descending-sorted array by the time p reaches 0.
        while (p > 0) {
            Idx k, kase;
            for (k = p - 2; k >= -1; --k) {
                if (k == -1) break;
                if (std::abs(e[S(k)]) <= tiny + eps * (std::abs(s_[S(k)]) + std::abs(s_[S(k + 1)]))) {
                    e[S(k)] = T{0};
                    break;
                }
            }
            if (k == p - 2) {
                kase = 4;
            } else {
                Idx ks;
                for (ks = p - 1; ks >= k; --ks) {
                    if (ks == k) break;
                    T t = (ks != p ? std::abs(e[S(ks)]) : T{0}) + (ks != k + 1 ? std::abs(e[S(ks - 1)]) : T{0});
                    if (std::abs(s_[S(ks)]) <= tiny + eps * t) {
                        s_[S(ks)] = T{0};
                        break;
                    }
                }
                if (ks == k) {
                    kase = 3;
                } else if (ks == p - 1) {
                    kase = 1;
                } else {
                    kase = 2;
                    k = ks;
                }
            }
            ++k;

            switch (kase) {
            case 1: {
                T f = e[S(p - 2)];
                e[S(p - 2)] = T{0};
                for (Idx j = p - 2; j >= k; --j) {
                    T t = std::hypot(s_[S(j)], f);
                    T cs = s_[S(j)] / t;
                    T sn = f / t;
                    s_[S(j)] = t;
                    if (j != k) {
                        f = -sn * e[S(j - 1)];
                        e[S(j - 1)] = cs * e[S(j - 1)];
                    }
                    for (std::size_t i = 0; i < n; ++i) {
                        T t2 = cs * V_(i, S(j)) + sn * V_(i, S(p - 1));
                        V_(i, S(p - 1)) = -sn * V_(i, S(j)) + cs * V_(i, S(p - 1));
                        V_(i, S(j)) = t2;
                    }
                }
                break;
            }
            case 2: {
                T f = e[S(k - 1)];
                e[S(k - 1)] = T{0};
                for (Idx j = k; j < p; ++j) {
                    T t = std::hypot(s_[S(j)], f);
                    T cs = s_[S(j)] / t;
                    T sn = f / t;
                    s_[S(j)] = t;
                    f = -sn * e[S(j)];
                    e[S(j)] = cs * e[S(j)];
                    for (std::size_t i = 0; i < m; ++i) {
                        T t2 = cs * U_(i, S(j)) + sn * U_(i, S(k - 1));
                        U_(i, S(k - 1)) = -sn * U_(i, S(j)) + cs * U_(i, S(k - 1));
                        U_(i, S(j)) = t2;
                    }
                }
                break;
            }
            case 3: {
                T scale = std::max({std::abs(s_[S(p - 1)]), std::abs(s_[S(p - 2)]), std::abs(e[S(p - 2)]),
                                     std::abs(s_[S(k)]), std::abs(e[S(k)])});
                T sp = s_[S(p - 1)] / scale;
                T spm1 = s_[S(p - 2)] / scale;
                T epm1 = e[S(p - 2)] / scale;
                T sk = s_[S(k)] / scale;
                T ek = e[S(k)] / scale;
                T b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / T{2};
                T c = (sp * epm1) * (sp * epm1);
                T shift = T{0};
                if (b != T{0} || c != T{0}) {
                    shift = std::sqrt(b * b + c);
                    if (b < T{0}) shift = -shift;
                    shift = c / (b + shift);
                }
                T f = (sk + sp) * (sk - sp) + shift;
                T g = sk * ek;
                for (Idx j = k; j < p - 1; ++j) {
                    T t = std::hypot(f, g);
                    T cs = f / t;
                    T sn = g / t;
                    if (j != k) e[S(j - 1)] = t;
                    f = cs * s_[S(j)] + sn * e[S(j)];
                    e[S(j)] = cs * e[S(j)] - sn * s_[S(j)];
                    g = sn * s_[S(j + 1)];
                    s_[S(j + 1)] = cs * s_[S(j + 1)];
                    for (std::size_t i = 0; i < n; ++i) {
                        T t2 = cs * V_(i, S(j)) + sn * V_(i, S(j + 1));
                        V_(i, S(j + 1)) = -sn * V_(i, S(j)) + cs * V_(i, S(j + 1));
                        V_(i, S(j)) = t2;
                    }
                    t = std::hypot(f, g);
                    cs = f / t;
                    sn = g / t;
                    s_[S(j)] = t;
                    f = cs * e[S(j)] + sn * s_[S(j + 1)];
                    s_[S(j + 1)] = -sn * e[S(j)] + cs * s_[S(j + 1)];
                    g = sn * e[S(j + 1)];
                    e[S(j + 1)] = cs * e[S(j + 1)];
                    if (j < M - 1) {
                        for (std::size_t i = 0; i < m; ++i) {
                            T t3 = cs * U_(i, S(j)) + sn * U_(i, S(j + 1));
                            U_(i, S(j + 1)) = -sn * U_(i, S(j)) + cs * U_(i, S(j + 1));
                            U_(i, S(j)) = t3;
                        }
                    }
                }
                e[S(p - 2)] = f;
                ++iter;
                break;
            }
            case 4: {
                if (s_[S(k)] <= T{0}) {
                    s_[S(k)] = (s_[S(k)] < T{0}) ? -s_[S(k)] : T{0};
                    for (Idx i = 0; i <= pp; ++i) V_(S(i), S(k)) = -V_(S(i), S(k));
                }
                while (k < pp) {
                    if (s_[S(k)] >= s_[S(k + 1)]) break;
                    std::swap(s_[S(k)], s_[S(k + 1)]);
                    if (k < N - 1) {
                        for (std::size_t i = 0; i < n; ++i) std::swap(V_(i, S(k + 1)), V_(i, S(k)));
                    }
                    if (k < M - 1) {
                        for (std::size_t i = 0; i < m; ++i) std::swap(U_(i, S(k + 1)), U_(i, S(k)));
                    }
                    ++k;
                }
                iter = 0;
                --p;
                break;
            }
            default:
                break;
            }
        }
        (void)iter;
    }

    Matrix<T> U_;
    Matrix<T> V_;
    std::vector<T> s_;
};

}  // namespace libquant
