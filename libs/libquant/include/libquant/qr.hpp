#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libquant/matrix.hpp"

namespace libquant {

// Householder QR factorization A = QR for an m x n matrix with m >= n
// (Golub & Van Loan, Algorithm 5.2.1). Q is accumulated explicitly as an
// m x m orthogonal matrix; R is upper triangular (zero below row n-1).
//
// Step k zeroes column k below the diagonal with a reflector
// H_k = I - 2 v v^T chosen so H_k R(k:m-1, k) = alpha * e_1. Explicit
// orthogonal triangularization is used -- rather than the normal equations
// A^T A x = A^T b -- because its error sensitivity is governed by kappa(A),
// not kappa(A)^2.
template <typename T = double>
class QR {
public:
    explicit QR(Matrix<T> A) : m_(A.rows()), n_(A.cols()), R_(std::move(A)) {
        if (m_ < n_) throw std::invalid_argument("QR: requires rows >= cols");
        Q_ = Matrix<T>::identity(m_);

        std::vector<T> v(m_, T{0});
        for (std::size_t k = 0; k < n_; ++k) {
            T norm_x = T{0};
            for (std::size_t i = k; i < m_; ++i) norm_x += R_(i, k) * R_(i, k);
            norm_x = std::sqrt(norm_x);
            if (norm_x == T{0}) continue;

            const T alpha = R_(k, k) >= T{0} ? -norm_x : norm_x;  // avoid cancellation
            for (std::size_t i = k; i < m_; ++i) v[i] = R_(i, k);
            v[k] -= alpha;
            T v_norm = T{0};
            for (std::size_t i = k; i < m_; ++i) v_norm += v[i] * v[i];
            v_norm = std::sqrt(v_norm);
            if (v_norm == T{0}) continue;
            for (std::size_t i = k; i < m_; ++i) v[i] /= v_norm;

            // Apply H_k = I - 2 v v^T to R(k:m-1, k:n-1).
            for (std::size_t j = k; j < n_; ++j) {
                T d = T{0};
                for (std::size_t i = k; i < m_; ++i) d += v[i] * R_(i, j);
                d *= T{2};
                for (std::size_t i = k; i < m_; ++i) R_(i, j) -= d * v[i];
            }
            // Accumulate Q := Q * H_k (H_k acts only on columns k..m-1 of Q).
            for (std::size_t i = 0; i < m_; ++i) {
                T d = T{0};
                for (std::size_t j = k; j < m_; ++j) d += Q_(i, j) * v[j];
                d *= T{2};
                for (std::size_t j = k; j < m_; ++j) Q_(i, j) -= d * v[j];
            }

            R_(k, k) = alpha;
            for (std::size_t i = k + 1; i < m_; ++i) R_(i, k) = T{0};
        }
    }

    [[nodiscard]] const Matrix<T>& Q() const noexcept { return Q_; }
    [[nodiscard]] const Matrix<T>& R() const noexcept { return R_; }

    // Least-squares solve of A x = b (full column rank): x = R1^{-1} (Q^T b)[0:n-1].
    [[nodiscard]] Matrix<T> solve(const Matrix<T>& b) const {
        if (b.rows() != m_) throw std::invalid_argument("QR::solve: rhs dimension mismatch");
        Matrix<T> qtb = Q_.transpose() * b;
        Matrix<T> x(n_, b.cols());
        for (std::size_t c = 0; c < b.cols(); ++c) {
            for (std::size_t ii = 0; ii < n_; ++ii) {
                std::size_t i = n_ - 1 - ii;
                T acc = qtb(i, c);
                for (std::size_t j = i + 1; j < n_; ++j) acc -= R_(i, j) * x(j, c);
                if (R_(i, i) == T{0}) throw std::runtime_error("QR::solve: rank deficient");
                x(i, c) = acc / R_(i, i);
            }
        }
        return x;
    }

private:
    std::size_t m_, n_;
    Matrix<T> Q_;
    Matrix<T> R_;
};

}  // namespace libquant
