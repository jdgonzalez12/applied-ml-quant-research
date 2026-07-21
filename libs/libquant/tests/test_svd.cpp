#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <random>

#include "libquant/matrix.hpp"
#include "libquant/svd.hpp"

using libquant::Matrix;
using libquant::SVD;
using Catch::Approx;

namespace {
Matrix<double> to_libquant(const Eigen::MatrixXd& Ae) {
    Matrix<double> A(static_cast<std::size_t>(Ae.rows()), static_cast<std::size_t>(Ae.cols()));
    for (int i = 0; i < Ae.rows(); ++i)
        for (int j = 0; j < Ae.cols(); ++j)
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = Ae(i, j);
    return A;
}
}  // namespace

TEST_CASE("SVD reconstructs a small known matrix and U, V are orthonormal", "[svd]") {
    const std::size_t m = 4, n = 3;
    Matrix<double> A(m, n);
    double v = 1.0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i)
            A(i, j) = (v++) * (((i + j) % 2 == 0) ? 1.0 : -1.0);

    SVD<double> svd(A);
    const auto& s = svd.singular_values();
    const auto& U = svd.U();
    const auto& V = svd.V();

    // Singular values are non-negative and descending.
    for (std::size_t i = 0; i < n; ++i) REQUIRE(s[i] >= -1e-9);
    for (std::size_t i = 0; i + 1 < n; ++i) REQUIRE(s[i] >= s[i + 1] - 1e-9);

    // Reconstruction: A == U * diag(s) * V^T.
    Matrix<double> Sigma(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) Sigma(i, i) = s[i];
    Matrix<double> recon = U * Sigma * V.transpose();
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(recon(i, j) == Approx(A(i, j)).margin(1e-8));

    // U^T U == I_n, V^T V == I_n.
    Matrix<double> UtU = U.transpose() * U;
    Matrix<double> VtV = V.transpose() * V;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(UtU(i, j) == Approx(i == j ? 1.0 : 0.0).margin(1e-8));
            REQUIRE(VtV(i, j) == Approx(i == j ? 1.0 : 0.0).margin(1e-8));
        }
}

TEST_CASE("SVD singular values cross-checked against Eigen on random matrices", "[svd][eigen]") {
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);

    for (auto [m, n] : std::vector<std::pair<std::size_t, std::size_t>>{{8, 5}, {6, 6}, {12, 3}}) {
        Eigen::MatrixXd Ae(static_cast<int>(m), static_cast<int>(n));
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                Ae(static_cast<int>(i), static_cast<int>(j)) = dist(rng);

        Matrix<double> A = to_libquant(Ae);
        SVD<double> svd(A);
        Eigen::JacobiSVD<Eigen::MatrixXd> esvd(Ae);
        Eigen::VectorXd es = esvd.singularValues();

        const auto& s = svd.singular_values();
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(s[i] == Approx(es(static_cast<int>(i))).margin(1e-6));
    }
}

TEST_CASE("SVD-based condition number matches Eigen on the Hilbert matrix", "[svd][eigen]") {
    const std::size_t n = 6;
    Eigen::MatrixXd He(static_cast<int>(n), static_cast<int>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            He(static_cast<int>(i), static_cast<int>(j)) = 1.0 / static_cast<double>(i + j + 1);

    Matrix<double> H = to_libquant(He);
    SVD<double> svd(H);
    const auto& s = svd.singular_values();
    double kappa = s.front() / s.back();

    Eigen::JacobiSVD<Eigen::MatrixXd> esvd(He);
    Eigen::VectorXd es = esvd.singularValues();
    double eigen_kappa = es(0) / es(static_cast<int>(n - 1));

    REQUIRE(kappa == Approx(eigen_kappa).epsilon(1e-4));
}
