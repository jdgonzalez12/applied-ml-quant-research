#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <random>

#include "libquant/matrix.hpp"
#include "libquant/symmetric_eigen.hpp"

using libquant::Matrix;
using libquant::SymmetricEigen;
using Catch::Approx;

TEST_CASE("SymmetricEigen recovers known eigenvalues of a diagonal matrix", "[symeig]") {
    Matrix<double> A(3, 3, 0.0);
    A(0, 0) = 5.0; A(1, 1) = 1.0; A(2, 2) = 3.0;

    SymmetricEigen<double> eig(A);
    const auto& vals = eig.eigenvalues();
    REQUIRE(vals[0] == Approx(5.0).margin(1e-9));
    REQUIRE(vals[1] == Approx(3.0).margin(1e-9));
    REQUIRE(vals[2] == Approx(1.0).margin(1e-9));
}

TEST_CASE("SymmetricEigen: Av = lambda v and eigenvectors are orthonormal, cross-checked vs Eigen", "[symeig][eigen]") {
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const std::size_t n = 6;

    Eigen::MatrixXd Be(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Be(static_cast<int>(i), static_cast<int>(j)) = dist(rng);
    Eigen::MatrixXd Ae = (Be + Be.transpose()) * 0.5;  // symmetrize

    Matrix<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = Ae(static_cast<int>(i), static_cast<int>(j));

    SymmetricEigen<double> eig(A);
    const auto& vals = eig.eigenvalues();
    const auto& V = eig.eigenvectors();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ae);
    std::vector<double> eigen_vals(n);
    for (std::size_t i = 0; i < n; ++i)
        eigen_vals[i] = solver.eigenvalues()(static_cast<int>(n - 1 - i));  // Eigen returns ascending; we want descending

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(vals[i] == Approx(eigen_vals[i]).margin(1e-7));

    // Av = lambda v for each recovered eigenpair (sign of v is not fixed).
    for (std::size_t j = 0; j < n; ++j) {
        Matrix<double> v(n, 1);
        for (std::size_t i = 0; i < n; ++i) v(i, 0) = V(i, j);
        Matrix<double> Av = A * v;
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(Av(i, 0) == Approx(vals[j] * v(i, 0)).margin(1e-6));
    }

    // Orthonormality: V^T V == I.
    Matrix<double> VtV = V.transpose() * V;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(VtV(i, j) == Approx(i == j ? 1.0 : 0.0).margin(1e-8));
}
