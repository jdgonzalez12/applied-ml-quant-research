#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <random>

#include "libquant/matrix.hpp"
#include "libquant/qr.hpp"

using libquant::Matrix;
using libquant::QR;
using Catch::Approx;

TEST_CASE("QR: Q is orthogonal and QR reconstructs A", "[qr]") {
    Matrix<double> A(4, 3);
    double v = 1.0;
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 4; ++i)
            A(i, j) = v++ * (i % 2 == 0 ? 1 : -1);

    QR<double> qr(A);
    const auto& Q = qr.Q();
    const auto& R = qr.R();

    Matrix<double> QtQ = Q.transpose() * Q;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE(QtQ(i, j) == Approx(i == j ? 1.0 : 0.0).margin(1e-9));

    Matrix<double> recon = Q * R;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(recon(i, j) == Approx(A(i, j)).margin(1e-9));

    // R must be upper triangular.
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = j + 1; i < 4; ++i)
            REQUIRE(R(i, j) == Approx(0.0).margin(1e-9));
}

TEST_CASE("QR least squares matches Eigen's least-squares solve", "[qr][eigen]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    const std::size_t m = 10, n = 4;

    Eigen::MatrixXd Ae(m, n);
    Matrix<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double val = dist(rng);
            A(i, j) = val;
            Ae(static_cast<int>(i), static_cast<int>(j)) = val;
        }
    Eigen::VectorXd be = Eigen::VectorXd::Random(static_cast<int>(m));
    Matrix<double> b(m, 1);
    for (std::size_t i = 0; i < m; ++i) b(i, 0) = be(static_cast<int>(i));

    QR<double> qr(A);
    Matrix<double> x = qr.solve(b);
    Eigen::VectorXd xe = Ae.colPivHouseholderQr().solve(be);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x(i, 0) == Approx(xe(static_cast<int>(i))).margin(1e-8));
}
