#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libquant/matrix.hpp"

namespace libquant {

// Cholesky factorization A = L L^T for symmetric positive-definite A
// (Golub & Van Loan, Algorithm 4.2.1). Roughly half the flops of LU (n^3/3
// vs 2n^3/3) because symmetry lets each column be produced from a single
// inner-product update instead of a full elimination step, and no pivoting
// is required for SPD matrices.
template <typename T = double>
class Cholesky {
public:
    explicit Cholesky(const Matrix<T>& A) : L_(A.rows(), A.cols(), T{0}) {
        if (!A.is_square()) throw std::invalid_argument("Cholesky: matrix must be square");
        const std::size_t n = A.rows();
        // Left-looking (gaxpy) column-oriented formulation (Golub & Van
        // Loan, Algorithm 4.2.1, column variant): each previous column k<j
        // contributes via a contiguous, unit-stride update to column j
        // scaled by a loop-invariant scalar -- the cache-friendly pattern
        // for column-major storage (the row-wise dot-product formulation
        // this replaced jumps `rows_` elements per step instead).
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = j; i < n; ++i) L_(i, j) = A(i, j);
            for (std::size_t k = 0; k < j; ++k) {
                const T ljk = L_(j, k);
                if (ljk == T{0}) continue;
                for (std::size_t i = j; i < n; ++i) L_(i, j) -= L_(i, k) * ljk;
            }
            if (L_(j, j) <= T{0}) { positive_definite_ = false; return; }
            L_(j, j) = std::sqrt(L_(j, j));
            const T inv_diag = T{1} / L_(j, j);
            for (std::size_t i = j + 1; i < n; ++i) L_(i, j) *= inv_diag;
        }
    }

    [[nodiscard]] bool is_positive_definite() const noexcept { return positive_definite_; }
    [[nodiscard]] const Matrix<T>& L() const noexcept { return L_; }

    // Solve Ax = b via L y = b (forward) then L^T x = y (backward).
    [[nodiscard]] Matrix<T> solve(const Matrix<T>& b) const {
        if (!positive_definite_) throw std::runtime_error("Cholesky::solve: matrix is not SPD");
        const std::size_t n = L_.rows();
        if (b.rows() != n) throw std::invalid_argument("Cholesky::solve: rhs dimension mismatch");

        Matrix<T> x(n, b.cols());
        std::vector<T> y(n);
        for (std::size_t c = 0; c < b.cols(); ++c) {
            for (std::size_t i = 0; i < n; ++i) {
                T acc = b(i, c);
                for (std::size_t k = 0; k < i; ++k) acc -= L_(i, k) * y[k];
                y[i] = acc / L_(i, i);
            }
            for (std::size_t ii = 0; ii < n; ++ii) {
                std::size_t i = n - 1 - ii;
                T acc = y[i];
                for (std::size_t k = i + 1; k < n; ++k) acc -= L_(k, i) * x(k, c);
                x(i, c) = acc / L_(i, i);
            }
        }
        return x;
    }

private:
    Matrix<T> L_;
    bool positive_definite_ = true;
};

}  // namespace libquant
