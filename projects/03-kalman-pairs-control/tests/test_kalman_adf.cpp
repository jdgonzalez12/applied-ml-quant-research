#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <random>

#include "adf_test.hpp"
#include "kalman_filter.hpp"

using Catch::Approx;
using namespace qmp03;

TEST_CASE("KalmanFilter1D converges to the true constant hedge ratio", "[kalman]") {
    std::mt19937 rng(42);
    std::normal_distribution<double> x_dist(1.0, 0.3);
    std::normal_distribution<double> noise(0.0, 0.01);
    const double beta_true = 1.4;

    KalmanFilter1D kf(0.0, 1.0, 1e-8, 0.01 * 0.01);
    double last_beta = 0.0;
    for (int t = 0; t < 500; ++t) {
        double xt = x_dist(rng);
        double yt = beta_true * xt + noise(rng);
        last_beta = kf.step(xt, yt).beta;
    }
    REQUIRE(last_beta == Approx(beta_true).margin(0.05));
}

TEST_CASE("KalmanFilter1D steady-state gain matches the closed-form fixed point", "[kalman]") {
    // For a constant regressor x and process/obs variances Q, R, the
    // steady-state prediction variance P satisfies the discrete Riccati
    // fixed point: P = ((P+Q) x^2 R) / (x^2 (P+Q) + R) ... solved here
    // numerically by iterating the filter with a fixed x until P stabilizes,
    // and checking the resulting Kalman gain K = (P+Q)x / (x^2(P+Q)+R)
    // against the same formula evaluated at the converged P.
    const double x = 1.0, Q = 1e-4, R = 0.02 * 0.02;
    KalmanFilter1D kf(0.0, 1.0, Q, R);
    double last_var = 1.0;
    for (int t = 0; t < 2000; ++t) {
        kf.step(x, 0.0);
        last_var = kf.variance();
    }
    double P_pred = last_var + Q;
    double S = x * x * P_pred + R;
    double K = P_pred * x / S;
    double P_next = (1.0 - K * x) * P_pred;
    // At steady state, one more predict+update step should leave P unchanged.
    REQUIRE(P_next == Approx(last_var).margin(1e-6));
}

TEST_CASE("adf_test rejects the unit-root null for a stationary AR(1) residual", "[adf]") {
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<double> u(500);
    u[0] = 0.0;
    for (std::size_t t = 1; t < u.size(); ++t) u[t] = 0.5 * u[t - 1] + noise(rng);  // stationary AR(1)

    ADFResult res = adf_test(u);
    REQUIRE(res.t_stat < res.critical_1pct);
    REQUIRE(res.cointegrated_5pct);
}

TEST_CASE("adf_test fails to reject the unit-root null for a pure random walk", "[adf]") {
    std::mt19937 rng(2);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<double> u(500);
    u[0] = 0.0;
    for (std::size_t t = 1; t < u.size(); ++t) u[t] = u[t - 1] + noise(rng);  // random walk: unit root

    ADFResult res = adf_test(u);
    REQUIRE(res.t_stat > res.critical_10pct);
    REQUIRE_FALSE(res.cointegrated_5pct);
}
