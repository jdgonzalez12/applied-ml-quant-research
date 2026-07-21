#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libquant/matrix.hpp"

namespace libquant {

// PA = LU factorization via Gaussian elimination with partial pivoting
// (Golub & Van Loan, Algorithm 3.4.1). L is unit lower triangular, U is
// upper triangular; both are packed into a single working matrix in place.
// P is tracked as a row-permutation vector rather than materialized.
template <typename T = double>
class LU {
public:
    explicit LU(Matrix<T> A) : lu_(std::move(A)) {
        if (!lu_.is_square()) throw std::invalid_argument("LU: matrix must be square");
        const std::size_t n = lu_.rows();
        piv_.resize(n);
        for (std::size_t i = 0; i < n; ++i) piv_[i] = i;

        for (std::size_t k = 0; k < n; ++k) {
            // Partial pivoting: pick the largest-magnitude entry in column k
            // among rows k..n-1. This bounds the growth factor and keeps
            // every multiplier |l_ij| <= 1, which is what makes the
            // algorithm backward stable in practice.
            std::size_t p = k;
            T best = std::abs(lu_(k, k));
            for (std::size_t i = k + 1; i < n; ++i) {
                T v = std::abs(lu_(i, k));
                if (v > best) { best = v; p = i; }
            }
            if (best == T{0}) { singular_ = true; continue; }
            if (p != k) {
                lu_.swap_rows(p, k);
                std::swap(piv_[p], piv_[k]);
                sign_ = -sign_;
            }
            // Rank-1 (outer-product) elimination, column by column: with
            // column-major storage, updating one destination column j at a
            // time keeps the inner loop over i a contiguous, unit-stride
            // sweep down that column -- the ijk/row-wise variant used
            // naively here previously made the inner loop jump `rows_`
            // elements per step, which is what caused the LU benchmark to
            // fall increasingly behind Eigen as n grew past cache size.
            const T pivot = lu_(k, k);
            for (std::size_t i = k + 1; i < n; ++i) lu_(i, k) /= pivot;  // multipliers, in place
            for (std::size_t j = k + 1; j < n; ++j) {
                const T ukj = lu_(k, j);
                if (ukj == T{0}) continue;
                for (std::size_t i = k + 1; i < n; ++i)
                    lu_(i, j) -= lu_(i, k) * ukj;
            }
        }
    }

    [[nodiscard]] bool singular() const noexcept { return singular_; }

    [[nodiscard]] T determinant() const {
        T d = sign_;
        for (std::size_t i = 0; i < lu_.rows(); ++i) d *= lu_(i, i);
        return d;
    }

    // Solve Ax = b (b may have multiple right-hand-side columns) via
    // forward substitution (Ly = Pb) then back substitution (Ux = y).
    [[nodiscard]] Matrix<T> solve(const Matrix<T>& b) const {
        const std::size_t n = lu_.rows();
        if (singular_) throw std::runtime_error("LU::solve: matrix is singular");
        if (b.rows() != n) throw std::invalid_argument("LU::solve: rhs dimension mismatch");

        Matrix<T> x(n, b.cols());
        std::vector<T> y(n);
        for (std::size_t c = 0; c < b.cols(); ++c) {
            for (std::size_t i = 0; i < n; ++i) {
                T acc = b(piv_[i], c);
                for (std::size_t j = 0; j < i; ++j) acc -= lu_(i, j) * y[j];
                y[i] = acc;
            }
            for (std::size_t ii = 0; ii < n; ++ii) {
                std::size_t i = n - 1 - ii;
                T acc = y[i];
                for (std::size_t j = i + 1; j < n; ++j) acc -= lu_(i, j) * x(j, c);
                x(i, c) = acc / lu_(i, i);
            }
        }
        return x;
    }

    [[nodiscard]] Matrix<T> lower() const {
        const std::size_t n = lu_.rows();
        Matrix<T> L = Matrix<T>::identity(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < i; ++j)
                L(i, j) = lu_(i, j);
        return L;
    }

    [[nodiscard]] Matrix<T> upper() const {
        const std::size_t n = lu_.rows();
        Matrix<T> U(n, n, T{0});
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j)
                U(i, j) = lu_(i, j);
        return U;
    }

    // Row i of PA equals row pivots()[i] of the original A.
    [[nodiscard]] const std::vector<std::size_t>& pivots() const noexcept { return piv_; }

private:
    Matrix<T> lu_;
    std::vector<std::size_t> piv_;
    T sign_ = T{1};
    bool singular_ = false;
};

}  // namespace libquant
