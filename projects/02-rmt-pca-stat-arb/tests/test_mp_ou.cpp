#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <random>

#include "marchenko_pastur.hpp"
#include "ou_signal.hpp"

using Catch::Approx;
using namespace qmp02;

TEST_CASE("Marchenko-Pastur filter finds no genuine factors in pure IID noise", "[mp]") {
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 1.0);
    const std::size_t T = 500, N = 10;

    libquant::Matrix<double> R(T, N);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t j = 0; j < N; ++j) R(t, j) = noise(rng);

    auto C = correlation_matrix(R);
    double q = static_cast<double>(N) / static_cast<double>(T);
    auto mp = marchenko_pastur_filter(C, q);

    REQUIRE(mp.eigenvalues.size() == N);
    // Trace of a correlation matrix is N; the MP-denoised trace should stay close to N too.
    double trace_raw = 0.0, trace_denoised = 0.0;
    for (double v : mp.eigenvalues) trace_raw += v;
    for (double v : mp.eigenvalues_denoised) trace_denoised += v;
    REQUIRE(trace_raw == Approx(static_cast<double>(N)).margin(1e-6));
    REQUIRE(trace_denoised == Approx(static_cast<double>(N)).margin(1e-6));

    // For pure noise, at most a small handful of eigenvalues should exceed
    // the MP edge purely by finite-sample chance.
    std::size_t above_edge = 0;
    for (double lam : mp.eigenvalues)
        if (lam > mp.lambda_plus) ++above_edge;
    REQUIRE(above_edge <= 2);
}

TEST_CASE("Marchenko-Pastur filter detects a strong common factor", "[mp]") {
    std::mt19937 rng(2);
    std::normal_distribution<double> noise(0.0, 0.3);
    std::normal_distribution<double> market(0.0, 1.0);
    const std::size_t T = 500, N = 10;

    libquant::Matrix<double> R(T, N);
    for (std::size_t t = 0; t < T; ++t) {
        double m = market(rng);
        for (std::size_t j = 0; j < N; ++j) R(t, j) = m + noise(rng);  // one shared market factor + idiosyncratic noise
    }

    auto C = correlation_matrix(R);
    double q = static_cast<double>(N) / static_cast<double>(T);
    auto mp = marchenko_pastur_filter(C, q);

    REQUIRE(mp.eigenvalues[0] > mp.lambda_plus);
    // The dominant eigenvalue should be far larger than the rest for a
    // strongly shared single-factor structure.
    REQUIRE(mp.eigenvalues[0] > mp.eigenvalues[1] * 2.0);
}

TEST_CASE("fit_ou recovers a known mean-reversion speed from a simulated AR(1)", "[ou]") {
    std::mt19937 rng(3);
    std::normal_distribution<double> noise(0.0, 0.1);
    const double b_true = 0.9;
    const double a_true = 0.05;

    std::vector<double> X(300);
    X[0] = 0.0;
    for (std::size_t t = 1; t < X.size(); ++t) X[t] = a_true + b_true * X[t - 1] + noise(rng);

    OUFit fit = fit_ou(X);
    REQUIRE(fit.valid);
    REQUIRE(fit.b == Approx(b_true).margin(0.05));
    double expected_half_life = std::log(2.0) / (-std::log(b_true));
    REQUIRE(fit.half_life == Approx(expected_half_life).margin(1.0));
}

TEST_CASE("s_score is zero at the OU equilibrium mean", "[ou]") {
    OUFit fit;
    fit.valid = true;
    fit.mean = 2.0;
    fit.sigma_eq = 0.5;
    REQUIRE(s_score(fit, 2.0) == Approx(0.0));
    REQUIRE(s_score(fit, 2.5) == Approx(1.0));
    REQUIRE(s_score(fit, 1.5) == Approx(-1.0));
}
