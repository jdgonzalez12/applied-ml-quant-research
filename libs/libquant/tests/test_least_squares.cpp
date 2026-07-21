#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <random>

#include "libquant/least_squares.hpp"
#include "libquant/matrix.hpp"

using libquant::Matrix;
using Catch::Approx;

TEST_CASE("ridge with lambda=0 matches OLS on a full-rank system", "[least_squares]") {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    const std::size_t m = 10, n = 4;

    Matrix<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = dist(rng);
    Matrix<double> b(m, 1);
    for (std::size_t i = 0; i < m; ++i) b(i, 0) = dist(rng);

    Matrix<double> x_ols = libquant::ols(A, b);
    Matrix<double> x_ridge0 = libquant::ridge(A, b, 0.0);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x_ridge0(i, 0) == Approx(x_ols(i, 0)).margin(1e-8));
}

TEST_CASE("ridge matches the closed-form normal-equation solution, cross-checked vs Eigen", "[least_squares][eigen]") {
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const std::size_t m = 8, n = 4;
    const double lambda = 2.5;

    Eigen::MatrixXd Ae(static_cast<int>(m), static_cast<int>(n));
    Matrix<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double v = dist(rng);
            A(i, j) = v;
            Ae(static_cast<int>(i), static_cast<int>(j)) = v;
        }
    Eigen::VectorXd be = Eigen::VectorXd::Random(static_cast<int>(m));
    Matrix<double> b(m, 1);
    for (std::size_t i = 0; i < m; ++i) b(i, 0) = be(static_cast<int>(i));

    Matrix<double> x = libquant::ridge(A, b, lambda);

    Eigen::MatrixXd normal = Ae.transpose() * Ae + lambda * Eigen::MatrixXd::Identity(static_cast<int>(n), static_cast<int>(n));
    Eigen::VectorXd xe = normal.ldlt().solve(Ae.transpose() * be);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x(i, 0) == Approx(xe(static_cast<int>(i))).margin(1e-6));
}
