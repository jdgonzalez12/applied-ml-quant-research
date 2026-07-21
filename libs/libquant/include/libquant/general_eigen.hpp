#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "libquant/matrix.hpp"

namespace libquant {

// Characteristic polynomial coefficients c[0..n] (c[0] = 1) of an n x n
// matrix A, such that det(lambda I - A) = lambda^n + c[1] lambda^{n-1} +
// ... + c[n], via the Faddeev-LeVerrier recursion (uses only matrix
// products and traces -- no pivoting, no shifts, entirely from-scratch):
//   M_1 = I;  c_k = -tr(A M_k)/k;  M_{k+1} = A M_k + c_k I.
template <typename T = double>
[[nodiscard]] std::vector<T> characteristic_polynomial(const Matrix<T>& A) {
    const std::size_t n = A.rows();
    std::vector<T> c(n + 1);
    c[0] = T{1};
    Matrix<T> Mk = Matrix<T>::identity(n);
    for (std::size_t k = 1; k <= n; ++k) {
        Matrix<T> AMk = A * Mk;
        c[k] = -AMk.trace() / static_cast<T>(k);
        if (k < n) {
            for (std::size_t i = 0; i < n; ++i) AMk(i, i) += c[k];
            Mk = AMk;
        }
    }
    return c;
}

// Durand-Kerner (Weierstrass) simultaneous iteration for all n roots of a
// monic real-coefficient polynomial p(x) = x^n + c[1] x^{n-1} + ... + c[n]:
//   z_i <- z_i - p(z_i) / prod_{j != i} (z_i - z_j).
// Converges quadratically once roots separate; handles complex-conjugate
// pairs naturally since the iteration is carried out in std::complex<T>
// throughout, unlike a real Hessenberg-QR eigensolver which needs explicit
// 2x2 block deflation for complex pairs.
template <typename T = double>
[[nodiscard]] std::vector<std::complex<T>> polynomial_roots(const std::vector<T>& c, int max_iter = 500,
                                                              T tol = static_cast<T>(1e-12)) {
    const std::size_t n = c.size() - 1;
    std::vector<std::complex<T>> roots(n);
    if (n == 0) return roots;

    T radius = T{1};
    for (std::size_t i = 1; i < c.size(); ++i) radius = std::max(radius, std::abs(c[i]));
    radius += T{1};

    const T pi = static_cast<T>(3.14159265358979323846);
    const std::complex<T> perturb(static_cast<T>(0.4), static_cast<T>(0.9));  // breaks initial symmetry
    for (std::size_t i = 0; i < n; ++i) {
        T angle = static_cast<T>(2) * pi * static_cast<T>(i) / static_cast<T>(n);
        roots[i] = radius * std::complex<T>(std::cos(angle), std::sin(angle)) + perturb * (radius * static_cast<T>(0.01));
    }

    auto eval = [&](std::complex<T> z) {
        std::complex<T> result(T{1}, T{0});
        for (std::size_t k = 1; k < c.size(); ++k) result = result * z + c[k];
        return result;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        T max_delta = T{0};
        for (std::size_t i = 0; i < n; ++i) {
            std::complex<T> denom(T{1}, T{0});
            for (std::size_t j = 0; j < n; ++j) {
                if (j != i) denom *= (roots[i] - roots[j]);
            }
            if (std::abs(denom) < tol) continue;
            std::complex<T> delta = eval(roots[i]) / denom;
            roots[i] -= delta;
            max_delta = std::max(max_delta, std::abs(delta));
        }
        if (max_delta < tol) break;
    }
    return roots;
}

// Eigenvalues of a general (possibly non-symmetric) small matrix, via
// Faddeev-LeVerrier + Durand-Kerner. Deliberately scoped to eigenvalues
// only (no eigenvectors) and to small n (this library uses it only for
// DMD's rank-truncated reduced operator, r <= ~10) -- a full real-Schur
// Hessenberg-QR eigensolver with explicit complex-pair deflation would
// handle larger, more general problems, but is out of scope here; see the
// project README for the reasoning.
template <typename T = double>
class GeneralEigen {
public:
    explicit GeneralEigen(const Matrix<T>& A) {
        auto c = characteristic_polynomial(A);
        eigenvalues_ = polynomial_roots(c);
    }

    [[nodiscard]] const std::vector<std::complex<T>>& eigenvalues() const noexcept { return eigenvalues_; }

private:
    std::vector<std::complex<T>> eigenvalues_;
};

}  // namespace libquant
