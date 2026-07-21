#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "libquant/matrix.hpp"

namespace libquant {

// Eigendecomposition of a real symmetric matrix A = V diag(lambda) V^T via
// Householder tridiagonalization (Golub & Van Loan, Algorithm 8.3.1)
// followed by the implicit-shift symmetric QL algorithm on the resulting
// tridiagonal form (the EISPACK/Golub & Van Loan Sec. 8.3 "implicit
// symmetric QR step", Wilkinson-shifted). Eigenvalues are returned in
// descending order, the convention PCA needs (largest variance first).
template <typename T = double>
class SymmetricEigen {
public:
    explicit SymmetricEigen(Matrix<T> A) : n_(A.rows()) {
        if (!A.is_square()) throw std::invalid_argument("SymmetricEigen: matrix must be square");
        V_ = std::move(A);
        d_.assign(n_, T{0});
        e_.assign(n_, T{0});
        if (n_ > 0) {
            tridiagonalize();
            diagonalize();
            sort_descending();
        }
    }

    [[nodiscard]] const std::vector<T>& eigenvalues() const noexcept { return d_; }
    [[nodiscard]] const Matrix<T>& eigenvectors() const noexcept { return V_; }

private:
    // Reduces the symmetric matrix V_ (overwritten in place) to tridiagonal
    // form (diagonal d_, off-diagonal e_), accumulating the orthogonal
    // transform into V_ itself. Golub & Van Loan, Algorithm 8.3.1 (Householder
    // reduction), following the classical tred2 formulation.
    void tridiagonalize() {
        for (std::size_t i = n_; i-- > 1;) {
            std::size_t l = i - 1;
            T h = T{0}, scale = T{0};
            if (l > 0) {
                for (std::size_t k = 0; k <= l; ++k) scale += std::abs(V_(i, k));
                if (scale == T{0}) {
                    e_[i] = V_(i, l);
                } else {
                    for (std::size_t k = 0; k <= l; ++k) {
                        V_(i, k) /= scale;
                        h += V_(i, k) * V_(i, k);
                    }
                    T f = V_(i, l);
                    T g = (f >= T{0}) ? -std::sqrt(h) : std::sqrt(h);
                    e_[i] = scale * g;
                    h -= f * g;
                    V_(i, l) = f - g;
                    f = T{0};
                    for (std::size_t j = 0; j <= l; ++j) {
                        V_(j, i) = V_(i, j) / h;
                        T gg = T{0};
                        for (std::size_t k = 0; k <= j; ++k) gg += V_(j, k) * V_(i, k);
                        for (std::size_t k = j + 1; k <= l; ++k) gg += V_(k, j) * V_(i, k);
                        e_[j] = gg / h;
                        f += e_[j] * V_(i, j);
                    }
                    T hh = f / (h + h);
                    for (std::size_t j = 0; j <= l; ++j) {
                        f = V_(i, j);
                        g = e_[j] - hh * f;
                        e_[j] = g;
                        for (std::size_t k = 0; k <= j; ++k) V_(j, k) -= (f * e_[k] + g * V_(i, k));
                    }
                }
            } else {
                e_[i] = V_(i, l);
            }
            d_[i] = h;
        }
        d_[0] = T{0};
        e_[0] = T{0};
        for (std::size_t i = 0; i < n_; ++i) {
            if (d_[i] != T{0} && i > 0) {
                std::size_t l = i - 1;
                for (std::size_t j = 0; j <= l; ++j) {
                    T g = T{0};
                    for (std::size_t k = 0; k <= l; ++k) g += V_(i, k) * V_(k, j);
                    for (std::size_t k = 0; k <= l; ++k) V_(k, j) -= g * V_(k, i);
                }
            }
            d_[i] = V_(i, i);
            V_(i, i) = T{1};
            if (i > 0) {
                for (std::size_t j = 0; j <= i - 1; ++j) {
                    V_(j, i) = T{0};
                    V_(i, j) = T{0};
                }
            }
        }
    }

    static T sign_(T a, T b) { return b >= T{0} ? std::abs(a) : -std::abs(a); }

    // Implicit-shift symmetric QL sweep on the tridiagonal (d_, e_),
    // accumulating eigenvectors into V_.
    void diagonalize() {
        for (std::size_t i = 1; i < n_; ++i) e_[i - 1] = e_[i];
        e_[n_ - 1] = T{0};

        for (std::size_t l = 0; l < n_; ++l) {
            int iter = 0;
            std::size_t m = l;
            do {
                for (m = l; m + 1 < n_; ++m) {
                    T dd = std::abs(d_[m]) + std::abs(d_[m + 1]);
                    if (std::abs(e_[m]) <= std::numeric_limits<T>::epsilon() * dd) break;
                }
                if (m != l) {
                    if (++iter == 50) throw std::runtime_error("SymmetricEigen: QL algorithm failed to converge");
                    T g = (d_[l + 1] - d_[l]) / (T{2} * e_[l]);
                    T r = std::hypot(g, T{1});
                    g = d_[m] - d_[l] + e_[l] / (g + sign_(r, g));
                    T s = T{1}, c = T{1}, p = T{0};
                    for (std::size_t idx = m; idx-- > l;) {
                        T f = s * e_[idx];
                        T b = c * e_[idx];
                        r = std::hypot(f, g);
                        e_[idx + 1] = r;
                        if (r == T{0}) {
                            d_[idx + 1] -= p;
                            e_[m] = T{0};
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d_[idx + 1] - p;
                        r = (d_[idx] - g) * s + T{2} * c * b;
                        p = s * r;
                        d_[idx + 1] = g + p;
                        g = c * r - b;
                        for (std::size_t k = 0; k < n_; ++k) {
                            T f2 = V_(k, idx + 1);
                            V_(k, idx + 1) = s * V_(k, idx) + c * f2;
                            V_(k, idx) = c * V_(k, idx) - s * f2;
                        }
                    }
                    d_[l] -= p;
                    e_[l] = g;
                    e_[m] = T{0};
                }
            } while (m != l);
        }
    }

    void sort_descending() {
        std::vector<std::size_t> idx(n_);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return d_[a] > d_[b]; });
        std::vector<T> d_sorted(n_);
        Matrix<T> V_sorted(n_, n_);
        for (std::size_t j = 0; j < n_; ++j) {
            d_sorted[j] = d_[idx[j]];
            for (std::size_t i = 0; i < n_; ++i) V_sorted(i, j) = V_(i, idx[j]);
        }
        d_ = std::move(d_sorted);
        V_ = std::move(V_sorted);
    }

    std::size_t n_;
    Matrix<T> V_;
    std::vector<T> d_, e_;
};

}  // namespace libquant
