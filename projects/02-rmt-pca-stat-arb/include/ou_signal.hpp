#pragma once

#include <cmath>
#include <vector>

#include "libquant/least_squares.hpp"
#include "libquant/matrix.hpp"

namespace qmp02 {

struct OUFit {
    double a = 0.0, b = 0.0;  // X_t = a + b X_{t-1} + noise
    double kappa = 0.0;       // -ln(b): mean-reversion speed per period
    double half_life = 0.0;   // ln(2)/kappa, in periods
    double mean = 0.0;        // equilibrium mean m = a/(1-b)
    double sigma_eq = 0.0;    // equilibrium stdev of X
    bool valid = false;       // false if b is outside (0,1): no stationary mean reversion
};

// Fits an Ornstein-Uhlenbeck process to a cumulative-residual series X via
// its AR(1) discretization X_t = a + b X_{t-1} + zeta_t (Avellaneda & Lee
// 2010, "Statistical Arbitrage in the U.S. Equities Market"): b = e^{-kappa
// dt} maps directly to a mean-reversion speed and half-life, and
// (mean, sigma_eq) give the equilibrium level and dispersion the s-score is
// measured against.
inline OUFit fit_ou(const std::vector<double>& X) {
    OUFit fit;
    const std::size_t n = X.size();
    if (n < 6) return fit;  // need enough points for a meaningful AR(1) fit

    libquant::Matrix<double> A(n - 1, 2), y(n - 1, 1);
    for (std::size_t t = 1; t < n; ++t) {
        A(t - 1, 0) = 1.0;
        A(t - 1, 1) = X[t - 1];
        y(t - 1, 0) = X[t];
    }
    libquant::Matrix<double> beta = libquant::ols(A, y);
    fit.a = beta(0, 0);
    fit.b = beta(1, 0);

    if (fit.b <= 0.0 || fit.b >= 1.0) return fit;  // not a valid stationary AR(1)

    fit.kappa = -std::log(fit.b);
    fit.half_life = std::log(2.0) / fit.kappa;
    fit.mean = fit.a / (1.0 - fit.b);

    double resid_var = 0.0;
    for (std::size_t t = 1; t < n; ++t) {
        double pred = fit.a + fit.b * X[t - 1];
        double e = X[t] - pred;
        resid_var += e * e;
    }
    resid_var /= static_cast<double>(n - 1 - 2);  // 2 fitted parameters
    fit.sigma_eq = std::sqrt(resid_var / (1.0 - fit.b * fit.b));
    fit.valid = fit.sigma_eq > 0.0;
    return fit;
}

// Current deviation of the cumulative residual from its OU equilibrium, in
// equilibrium standard deviations.
inline double s_score(const OUFit& fit, double X_t) {
    if (!fit.valid || fit.sigma_eq <= 0.0) return 0.0;
    return (X_t - fit.mean) / fit.sigma_eq;
}

}  // namespace qmp02
