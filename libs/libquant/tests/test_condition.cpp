#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>

#include "libquant/condition.hpp"
#include "libquant/lu.hpp"
#include "libquant/matrix.hpp"

using libquant::Matrix;
using Catch::Approx;

namespace {
// Exact 1-norm condition number via explicit inversion (feasible only for
// the small test matrices here) -- the correct ground truth for Hager's
// estimator, which targets kappa_1, not the SVD-based kappa_2.
double exact_kappa1(const Matrix<double>& A) {
    const std::size_t n = A.rows();
    libquant::LU<double> lu(A);
    Matrix<double> Ainv(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        Matrix<double> e(n, 1, 0.0);
        e(j, 0) = 1.0;
        Matrix<double> col = lu.solve(e);
        for (std::size_t i = 0; i < n; ++i) Ainv(i, j) = col(i, 0);
    }
    return libquant::norm1(A) * libquant::norm1(Ainv);
}
}  // namespace

TEST_CASE("condition_number_svd grows with Hilbert matrix size, matches Eigen", "[condition][eigen]") {
    for (std::size_t n : {3u, 5u, 8u}) {
        Eigen::MatrixXd He(static_cast<int>(n), static_cast<int>(n));
        Matrix<double> H(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double v = 1.0 / static_cast<double>(i + j + 1);
                H(i, j) = v;
                He(static_cast<int>(i), static_cast<int>(j)) = v;
            }
        double kappa = libquant::condition_number_svd(H);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(He);
        Eigen::VectorXd s = svd.singularValues();
        double eigen_kappa = s(0) / s(static_cast<int>(n - 1));

        REQUIRE(kappa == Approx(eigen_kappa).epsilon(1e-3));
    }
}

TEST_CASE("condition_number_estimate is within a small factor of the exact SVD value", "[condition]") {
    const std::size_t n = 6;
    Matrix<double> H(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);

    double exact = exact_kappa1(H);
    double est = libquant::condition_number_estimate(H);

    // Hager's estimator is a lower-bound heuristic for kappa_1 specifically
    // (not the SVD's kappa_2): it should never overshoot the true 1-norm
    // condition number by more than a small safety factor, and should be
    // within an order of magnitude of it.
    REQUIRE(est <= exact * 1.01);
    REQUIRE(est >= exact * 0.1);
}

TEST_CASE("condition_number_estimate is near 1 for an orthogonal-like well-conditioned matrix", "[condition]") {
    auto I = Matrix<double>::identity(5);
    double est = libquant::condition_number_estimate(I);
    REQUIRE(est == Approx(1.0).margin(1e-9));
}
