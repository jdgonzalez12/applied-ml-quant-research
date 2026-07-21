#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <random>

#include "libquant/cholesky.hpp"
#include "libquant/matrix.hpp"

using libquant::Cholesky;
using libquant::Matrix;
using Catch::Approx;

TEST_CASE("Cholesky rejects a non-SPD matrix", "[cholesky]") {
    Matrix<double> A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 1;  // eigenvalues 3 and -1: not SPD

    Cholesky<double> chol(A);
    REQUIRE_FALSE(chol.is_positive_definite());
}

TEST_CASE("Cholesky reconstructs A = L L^T and solves Ax=b, cross-checked vs Eigen", "[cholesky][eigen]") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const std::size_t n = 5;

    // Build a random SPD matrix as B B^T + n*I.
    Eigen::MatrixXd Be(n, n);
    Matrix<double> B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double v = dist(rng);
            B(i, j) = v;
            Be(static_cast<int>(i), static_cast<int>(j)) = v;
        }
    Eigen::MatrixXd Ae = Be * Be.transpose() + static_cast<double>(n) * Eigen::MatrixXd::Identity(static_cast<int>(n), static_cast<int>(n));

    Matrix<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = Ae(static_cast<int>(i), static_cast<int>(j));

    Cholesky<double> chol(A);
    REQUIRE(chol.is_positive_definite());

    // Reconstruction check: L L^T == A.
    const Matrix<double>& L = chol.L();
    Matrix<double> recon = L * L.transpose();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(recon(i, j) == Approx(A(i, j)).margin(1e-8));

    // Solve cross-check vs Eigen's LLT.
    Eigen::VectorXd be = Eigen::VectorXd::Random(static_cast<int>(n));
    Matrix<double> b(n, 1);
    for (std::size_t i = 0; i < n; ++i) b(i, 0) = be(static_cast<int>(i));

    Matrix<double> x = chol.solve(b);
    Eigen::VectorXd xe = Ae.llt().solve(be);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x(i, 0) == Approx(xe(static_cast<int>(i))).margin(1e-8));
}
