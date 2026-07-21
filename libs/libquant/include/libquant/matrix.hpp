#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace libquant {

// Dense matrix, column-major contiguous storage: column j occupies
// data_[j*rows_ .. j*rows_+rows_), so column access and rank-1 updates are
// cache-sequential. Every decomposition in this library (LU, Cholesky, QR,
// SVD) is written to walk memory in this order.
template <typename T = double>
class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, T fill = T{0})
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    static Matrix identity(std::size_t n) {
        Matrix I(n, n, T{0});
        for (std::size_t i = 0; i < n; ++i) I(i, i) = T{1};
        return I;
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool is_square() const noexcept { return rows_ == cols_; }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    T& operator()(std::size_t i, std::size_t j) {
        assert(i < rows_ && j < cols_);
        return data_[j * rows_ + i];
    }
    const T& operator()(std::size_t i, std::size_t j) const {
        assert(i < rows_ && j < cols_);
        return data_[j * rows_ + i];
    }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Pointer to the start of column j: contiguous, length rows().
    T* col(std::size_t j) noexcept { return data_.data() + j * rows_; }
    const T* col(std::size_t j) const noexcept { return data_.data() + j * rows_; }

    [[nodiscard]] Matrix transpose() const {
        Matrix out(cols_, rows_);
        for (std::size_t j = 0; j < cols_; ++j)
            for (std::size_t i = 0; i < rows_; ++i)
                out(j, i) = (*this)(i, j);
        return out;
    }

    void swap_rows(std::size_t r1, std::size_t r2) {
        if (r1 == r2) return;
        for (std::size_t j = 0; j < cols_; ++j)
            std::swap((*this)(r1, j), (*this)(r2, j));
    }

    void swap_cols(std::size_t c1, std::size_t c2) {
        if (c1 == c2) return;
        T* a = col(c1);
        T* b = col(c2);
        for (std::size_t i = 0; i < rows_; ++i) std::swap(a[i], b[i]);
    }

    Matrix operator+(const Matrix& o) const { return elementwise(o, std::plus<T>{}); }
    Matrix operator-(const Matrix& o) const { return elementwise(o, std::minus<T>{}); }

    Matrix operator*(const Matrix& o) const {
        if (cols_ != o.rows_) throw std::invalid_argument("Matrix::operator*: dimension mismatch");
        Matrix out(rows_, o.cols_, T{0});
        // ikj loop order: the innermost loop strides down a contiguous
        // output column, matching the column-major layout above.
        for (std::size_t k = 0; k < cols_; ++k) {
            for (std::size_t j = 0; j < o.cols_; ++j) {
                const T scale = o(k, j);
                if (scale == T{0}) continue;
                T* out_col = out.col(j);
                const T* this_col = col(k);
                for (std::size_t i = 0; i < rows_; ++i)
                    out_col[i] += this_col[i] * scale;
            }
        }
        return out;
    }

    Matrix operator*(T scalar) const {
        Matrix out(*this);
        for (auto& v : out.data_) v *= scalar;
        return out;
    }

    [[nodiscard]] T frobenius_norm() const {
        T acc{0};
        for (const auto& v : data_) acc += v * v;
        return std::sqrt(acc);
    }

    [[nodiscard]] T trace() const {
        assert(is_square());
        T acc{0};
        for (std::size_t i = 0; i < rows_; ++i) acc += (*this)(i, i);
        return acc;
    }

private:
    template <typename BinOp>
    Matrix elementwise(const Matrix& o, BinOp op) const {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument("Matrix elementwise op: dimension mismatch");
        Matrix out(rows_, cols_);
        for (std::size_t idx = 0; idx < data_.size(); ++idx)
            out.data_[idx] = op(data_[idx], o.data_[idx]);
        return out;
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> data_;
};

// A column vector is modeled as an n x 1 Matrix<T>.
template <typename T = double>
using Vector = Matrix<T>;

template <typename T>
[[nodiscard]] T dot(const Matrix<T>& a, const Matrix<T>& b) {
    if (a.cols() != 1 || b.cols() != 1 || a.rows() != b.rows())
        throw std::invalid_argument("dot: expects column vectors of equal length");
    T acc{0};
    for (std::size_t i = 0; i < a.rows(); ++i) acc += a(i, 0) * b(i, 0);
    return acc;
}

template <typename T>
[[nodiscard]] T norm2(const Matrix<T>& v) {
    return std::sqrt(dot(v, v));
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j)
            os << m(i, j) << (j + 1 < m.cols() ? " " : "");
        os << "\n";
    }
    return os;
}

}  // namespace libquant
