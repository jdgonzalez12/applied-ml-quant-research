#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "libquant/lu.hpp"
#include "libquant/matrix.hpp"
#include "libquant/svd.hpp"

namespace libquant {

// Exact 2-norm condition number kappa_2(A) = sigma_max / sigma_min via the
// full SVD (Golub & Van Loan, Sec. 2.7), O(n^3). Used as ground truth
// against which the cheaper Hager estimator below is checked, and directly
// where an exact κ(A) is worth the cost.
template <typename T = double>
[[nodiscard]] T condition_number_svd(const Matrix<T>& A) {
    SVD<T> svd(A);
    const auto& s = svd.singular_values();
    if (s.back() == T{0}) return std::numeric_limits<T>::infinity();
    return s.front() / s.back();
}

// 1-norm matrix norm: max absolute column sum.
template <typename T = double>
[[nodiscard]] T norm1(const Matrix<T>& A) {
    T best = T{0};
    for (std::size_t j = 0; j < A.cols(); ++j) {
        T sum = T{0};
        for (std::size_t i = 0; i < A.rows(); ++i) sum += std::abs(A(i, j));
        best = std::max(best, sum);
    }
    return best;
}

// Hager's 1-norm condition estimator (Golub & Van Loan, Sec. 3.5.4; Hager
// 1984). Estimates kappa_1(A) = ||A||_1 * ||A^{-1}||_1 using only a handful
// of solves with A and A^T against one shared LU factorization -- O(n^2)
// per iteration versus the SVD's O(n^3) -- without ever forming A^{-1}.
template <typename T = double>
[[nodiscard]] T condition_number_estimate(const Matrix<T>& A) {
    if (!A.is_square()) throw std::invalid_argument("condition_number_estimate: matrix must be square");
    const std::size_t n = A.rows();

    LU<T> lu(A);
    if (lu.singular()) return std::numeric_limits<T>::infinity();
    LU<T> lut(A.transpose());

    Matrix<T> x(n, 1, T{1} / static_cast<T>(n));
    T gamma_prev = T{-1};
    T gamma = T{0};
    const std::size_t max_iter = std::min<std::size_t>(n + 5, 100);

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        Matrix<T> y = lu.solve(x);
        gamma = T{0};
        for (std::size_t i = 0; i < n; ++i) gamma += std::abs(y(i, 0));
        if (gamma <= gamma_prev) break;
        gamma_prev = gamma;

        Matrix<T> xi(n, 1);
        for (std::size_t i = 0; i < n; ++i) xi(i, 0) = y(i, 0) >= T{0} ? T{1} : T{-1};
        Matrix<T> z = lut.solve(xi);

        std::size_t j = 0;
        T zmax = std::abs(z(0, 0));
        for (std::size_t i = 1; i < n; ++i) {
            if (std::abs(z(i, 0)) > zmax) { zmax = std::abs(z(i, 0)); j = i; }
        }
        T zx = T{0};
        for (std::size_t i = 0; i < n; ++i) zx += z(i, 0) * x(i, 0);
        if (zmax <= zx) break;

        x = Matrix<T>(n, 1, T{0});
        x(j, 0) = T{1};
    }

    return norm1(A) * gamma;
}

}  // namespace libquant
