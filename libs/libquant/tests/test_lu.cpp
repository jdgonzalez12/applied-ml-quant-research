#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <random>

#include "libquant/lu.hpp"
#include "libquant/matrix.hpp"

using libquant::LU;
using libquant::Matrix;
using Catch::Approx;

TEST_CASE("LU solves a known 3x3 system exactly", "[lu]") {
    // A x = b with A chosen so that x = [1, 2, 3] exactly, and requiring a
    // row swap under partial pivoting (a11 is not the largest in column 0).
    Matrix<double> A(3, 3);
    A(0, 0) = 2; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 4; A(1, 1) = 3; A(1, 2) = 3;
    A(2, 0) = 8; A(2, 1) = 7; A(2, 2) = 9;

    Matrix<double> x_true(3, 1);
    x_true(0, 0) = 1; x_true(1, 0) = 2; x_true(2, 0) = 3;

    Matrix<double> b = A * x_true;

    LU<double> lu(A);
    REQUIRE_FALSE(lu.singular());
    Matrix<double> x = lu.solve(b);

    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(x(i, 0) == Approx(x_true(i, 0)).margin(1e-9));
}

TEST_CASE("LU detects a singular matrix", "[lu]") {
    Matrix<double> A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 4;  // row 2 = 2 * row 1

    LU<double> lu(A);
    REQUIRE(lu.singular());
}

TEST_CASE("LU determinant and solve cross-checked against Eigen on random matrices", "[lu][eigen]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int trial = 0; trial < 20; ++trial) {
        const std::size_t n = 6;
        Matrix<double> A(n, n);
        Eigen::MatrixXd Ae(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double v = dist(rng);
                A(i, j) = v;
                Ae(static_cast<int>(i), static_cast<int>(j)) = v;
            }
        // Nudge the diagonal to keep the matrix well away from singular.
        for (std::size_t i = 0; i < n; ++i) { A(i, i) += 10.0; Ae(static_cast<int>(i), static_cast<int>(i)) += 10.0; }

        Eigen::VectorXd be = Eigen::VectorXd::Random(static_cast<int>(n));
        Matrix<double> b(n, 1);
        for (std::size_t i = 0; i < n; ++i) b(i, 0) = be(static_cast<int>(i));

        LU<double> lu(A);
        REQUIRE_FALSE(lu.singular());
        Matrix<double> x = lu.solve(b);
        Eigen::VectorXd xe = Ae.partialPivLu().solve(be);

        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(x(i, 0) == Approx(xe(static_cast<int>(i))).margin(1e-8));

        REQUIRE(lu.determinant() == Approx(Ae.determinant()).epsilon(1e-6));
    }
}
