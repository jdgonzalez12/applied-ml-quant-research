#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <random>

#include "libquant/general_eigen.hpp"
#include "libquant/matrix.hpp"

using libquant::GeneralEigen;
using libquant::Matrix;
using Catch::Approx;

TEST_CASE("characteristic_polynomial matches a hand-computed 2x2 example", "[general_eigen]") {
    Matrix<double> A(2, 2, 0.0);
    A(0, 0) = 2; A(1, 1) = 3;  // diag(2,3): char poly = lambda^2 - 5 lambda + 6
    auto c = libquant::characteristic_polynomial(A);
    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(-5.0));
    REQUIRE(c[2] == Approx(6.0));
}

TEST_CASE("GeneralEigen recovers real eigenvalues of a diagonal matrix", "[general_eigen]") {
    Matrix<double> A(3, 3, 0.0);
    A(0, 0) = 5; A(1, 1) = -2; A(2, 2) = 0.5;
    GeneralEigen<double> eig(A);
    std::vector<double> re;
    for (auto& z : eig.eigenvalues()) re.push_back(z.real());
    std::sort(re.begin(), re.end());
    REQUIRE(re[0] == Approx(-2.0).margin(1e-6));
    REQUIRE(re[1] == Approx(0.5).margin(1e-6));
    REQUIRE(re[2] == Approx(5.0).margin(1e-6));
}

TEST_CASE("GeneralEigen recovers a known complex-conjugate pair (rotation matrix)", "[general_eigen]") {
    // A 2D rotation-scaling matrix [[a,-b],[b,a]] has eigenvalues a +/- bi.
    const double a = 0.8, b = 0.6;
    Matrix<double> A(2, 2);
    A(0, 0) = a; A(0, 1) = -b;
    A(1, 0) = b; A(1, 1) = a;

    GeneralEigen<double> eig(A);
    REQUIRE(eig.eigenvalues().size() == 2);
    for (auto& z : eig.eigenvalues()) {
        REQUIRE(std::abs(z.real() - a) < 1e-6);
        REQUIRE(std::abs(std::abs(z.imag()) - b) < 1e-6);
    }
}

TEST_CASE("GeneralEigen cross-checked against Eigen::EigenSolver on random matrices", "[general_eigen][eigen]") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    for (int trial = 0; trial < 10; ++trial) {
        const std::size_t n = 5;
        Eigen::MatrixXd Ae(static_cast<int>(n), static_cast<int>(n));
        Matrix<double> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double v = dist(rng);
                A(i, j) = v;
                Ae(static_cast<int>(i), static_cast<int>(j)) = v;
            }

        GeneralEigen<double> eig(A);
        Eigen::EigenSolver<Eigen::MatrixXd> esolver(Ae);
        auto eigen_vals = esolver.eigenvalues();

        // Compare sorted-by-real-part, then by imaginary part, magnitude sums
        // (avoids needing to solve the assignment/matching problem exactly).
        std::vector<std::complex<double>> ours(eig.eigenvalues().begin(), eig.eigenvalues().end());
        std::vector<std::complex<double>> theirs;
        for (int i = 0; i < eigen_vals.size(); ++i) theirs.push_back(eigen_vals(i));

        auto by_parts = [](const std::complex<double>& x, const std::complex<double>& y) {
            if (x.real() != y.real()) return x.real() < y.real();
            return x.imag() < y.imag();
        };
        std::sort(ours.begin(), ours.end(), by_parts);
        std::sort(theirs.begin(), theirs.end(), by_parts);

        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(ours[i].real() == Approx(theirs[i].real()).margin(1e-5));
            REQUIRE(std::abs(ours[i].imag()) == Approx(std::abs(theirs[i].imag())).margin(1e-5));
        }
    }
}
