#pragma once

#include "libquant/matrix.hpp"
#include "libquant/qr.hpp"
#include "libquant/svd.hpp"

namespace libquant {

// Ordinary least squares via Householder QR (see QR::solve): well-suited
// when A is full column rank and no regularization is needed.
template <typename T = double>
[[nodiscard]] Matrix<T> ols(const Matrix<T>& A, const Matrix<T>& b) {
    QR<T> qr(A);
    return qr.solve(b);
}

// Ridge regression, x = argmin ||Ax - b||^2 + lambda ||x||^2, via the SVD
// closed form (Golub & Van Loan, Sec. 5.5 / Hastie-Tibshirani-Friedman,
// Elements of Statistical Learning Sec. 3.4):
//   A = U diag(s) V^T  =>  x = V diag(s / (s^2 + lambda)) U^T b.
// Unlike the normal-equation form x = (A^T A + lambda I)^{-1} A^T b, this
// never squares kappa(A), and lambda = 0 degenerates to the minimum-norm
// least-squares solution.
template <typename T = double>
[[nodiscard]] Matrix<T> ridge(const Matrix<T>& A, const Matrix<T>& b, T lambda) {
    SVD<T> svd(A);
    const auto& s = svd.singular_values();
    const Matrix<T>& U = svd.U();
    const Matrix<T>& V = svd.V();
    const std::size_t n = A.cols();

    Matrix<T> utb = U.transpose() * b;  // n x c
    Matrix<T> scaled(n, b.cols());
    for (std::size_t i = 0; i < n; ++i) {
        const T factor = s[i] / (s[i] * s[i] + lambda);
        for (std::size_t c = 0; c < b.cols(); ++c) scaled(i, c) = factor * utb(i, c);
    }
    return V * scaled;
}

}  // namespace libquant
