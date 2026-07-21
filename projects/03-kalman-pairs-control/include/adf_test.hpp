#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "libquant/least_squares.hpp"
#include "libquant/matrix.hpp"

namespace qmp03 {

struct ADFResult {
    double gamma = 0.0;
    double t_stat = 0.0;
    double critical_1pct = -3.90;
    double critical_5pct = -3.34;
    double critical_10pct = -3.04;
    bool cointegrated_5pct = false;
};

// Engle-Granger two-step cointegration test, step 2: an augmented
// Dickey-Fuller regression (one lag) on the step-1 OLS residual series u_t,
//   Delta u_t = gamma * u_{t-1} + phi * Delta u_{t-1} + e_t
// (no constant/trend: u_t is already the residual of a regression that
// included one). The null hypothesis gamma = 0 is a unit root (no
// cointegration); rejecting in favor of gamma < 0 indicates the residual is
// stationary, i.e. the two series are cointegrated.
//
// Critical values are the asymptotic MacKinnon (1994/2010) values for the
// 2-variable, no-constant Engle-Granger residual test, hard-coded rather
// than computed from MacKinnon's full finite-sample response-surface
// regression -- a deliberate scope decision (see project README): deriving
// the asymptotic Dickey-Fuller distribution itself is out of scope.
inline ADFResult adf_test(const std::vector<double>& u) {
    ADFResult res;
    const std::size_t n = u.size();
    if (n < 10) return res;

    std::vector<double> du(n - 1);
    for (std::size_t t = 1; t < n; ++t) du[t - 1] = u[t] - u[t - 1];

    const std::size_t m = du.size() - 1;  // one lag of du consumed
    libquant::Matrix<double> X(m, 2), y(m, 1);
    for (std::size_t i = 0; i < m; ++i) {
        X(i, 0) = u[i + 1];   // u_{t-1} (level)
        X(i, 1) = du[i];      // Delta u_{t-1}
        y(i, 0) = du[i + 1];  // Delta u_t
    }

    libquant::Matrix<double> beta = libquant::ols(X, y);
    res.gamma = beta(0, 0);

    libquant::Matrix<double> resid = y - X * beta;
    double rss = 0.0;
    for (std::size_t i = 0; i < m; ++i) rss += resid(i, 0) * resid(i, 0);
    double sigma2 = rss / static_cast<double>(m - 2);

    libquant::Matrix<double> XtX = X.transpose() * X;
    double det = XtX(0, 0) * XtX(1, 1) - XtX(0, 1) * XtX(1, 0);
    double var_gamma = (det != 0.0) ? sigma2 * XtX(1, 1) / det : 0.0;
    double se_gamma = std::sqrt(std::max(var_gamma, 0.0));

    res.t_stat = (se_gamma > 0.0) ? res.gamma / se_gamma : 0.0;
    res.cointegrated_5pct = res.t_stat < res.critical_5pct;
    return res;
}

}  // namespace qmp03
