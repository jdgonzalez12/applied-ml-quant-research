// Project 3: Kalman-filter pairs trading vs. a static-OLS baseline.
//
// Pipeline (see README.md for the full derivation):
//   1. Engle-Granger cointegration test on log(CVX) ~ log(XOM).
//   2. Static hedge ratio: one OLS fit on an initial 252-day training
//      window, held fixed for the entire remainder of history.
//   3. Kalman-filter hedge ratio: beta_t updated every day via the
//      minimum-variance recursive filter, same random-walk state model.
//   4. Both produce a spread and rolling z-score signal, traded with the
//      identical hysteresis rule and backtest engine, over the identical
//      out-of-sample period, so the comparison isolates the value of
//      time-varying estimation.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "libquant/least_squares.hpp"
#include "libquant/matrix.hpp"
#include "libquant_backtest/metrics.hpp"

#include "adf_test.hpp"
#include "csv_loader.hpp"
#include "kalman_filter.hpp"

#ifndef QMP_DATA_DIR
#define QMP_DATA_DIR "data/raw/equities"
#endif
#ifndef QMP_RESULTS_DIR
#define QMP_RESULTS_DIR "results"
#endif

namespace {

double rolling_mean(const std::vector<double>& v, std::size_t end, std::size_t window) {
    double s = 0.0;
    for (std::size_t i = end - window; i < end; ++i) s += v[i];
    return s / static_cast<double>(window);
}
double rolling_std(const std::vector<double>& v, std::size_t end, std::size_t window, double mean) {
    double s = 0.0;
    for (std::size_t i = end - window; i < end; ++i) {
        double d = v[i] - mean;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(window - 1));
}

struct MethodState {
    std::vector<double> position_prev = {0.0};
};

}  // namespace

int main() {
    using namespace qmp03;

    std::cout << "Loading XOM/CVX price panel...\n";
    PricePanel panel = load_price_panel(QMP_DATA_DIR, {"xom", "cvx"});
    const std::size_t T = panel.dates.size();
    std::cout << "  " << T << " aligned trading days.\n";

    std::vector<double> x(T), y(T);  // log prices: x = log(XOM), y = log(CVX)
    for (std::size_t t = 0; t < T; ++t) {
        x[t] = std::log(panel.prices(t, 0));
        y[t] = std::log(panel.prices(t, 1));
    }

    const std::size_t train_window = 252;

    // Static OLS hedge ratio, fit once on the first `train_window` days,
    // held fixed for the rest of history (Golub & Van Loan QR-based LS).
    libquant::Matrix<double> Xtrain(train_window, 1), ytrain(train_window, 1);
    for (std::size_t t = 0; t < train_window; ++t) {
        Xtrain(t, 0) = x[t];
        ytrain(t, 0) = y[t];
    }
    libquant::Matrix<double> beta_static_mat = libquant::ols(Xtrain, ytrain);
    const double beta_static = beta_static_mat(0, 0);

    double train_resid_var = 0.0;
    for (std::size_t t = 0; t < train_window; ++t) {
        double e = y[t] - beta_static * x[t];
        train_resid_var += e * e;
    }
    train_resid_var /= static_cast<double>(train_window - 1);

    std::cout << "Static hedge ratio (252d training window): beta = " << beta_static << "\n";

    // Cointegration check on the full-sample static-beta spread.
    std::vector<double> full_spread(T);
    for (std::size_t t = 0; t < T; ++t) full_spread[t] = y[t] - beta_static * x[t];
    ADFResult adf = adf_test(full_spread);
    std::cout << "Engle-Granger ADF t-stat: " << adf.t_stat << " (5% critical value " << adf.critical_5pct
              << ", cointegrated=" << (adf.cointegrated_5pct ? "yes" : "no") << ")\n";

    // Kalman filter: initialized from the same training window, then run
    // (and only evaluated for trading) on the out-of-sample period.
    const double Q_process = 1e-7;  // beta random-walk variance per day: small, beta drifts slowly
    KalmanFilter1D kf(beta_static, 1e-2, Q_process, train_resid_var);

    std::vector<double> beta_kalman(T, beta_static);
    for (std::size_t t = 0; t < T; ++t) {
        auto step = kf.step(x[t], y[t]);
        beta_kalman[t] = step.beta;
    }

    std::filesystem::create_directories(QMP_RESULTS_DIR);
    std::ofstream beta_out(std::string(QMP_RESULTS_DIR) + "/beta_series.csv");
    beta_out << "date,beta_kalman,beta_static\n";
    for (std::size_t t = 0; t < T; ++t) beta_out << panel.dates[t] << "," << beta_kalman[t] << "," << beta_static << "\n";

    // Spreads for both methods, full series (rolling z-score needs history).
    std::vector<double> spread_kalman(T), spread_static(T);
    for (std::size_t t = 0; t < T; ++t) {
        spread_kalman[t] = y[t] - beta_kalman[t] * x[t];
        spread_static[t] = y[t] - beta_static * x[t];
    }

    const std::size_t zscore_window = 20;
    const double entry_z = 2.0, exit_z = 0.5;
    const double cost_one_way = 5e-4;

    auto run_backtest = [&](const std::vector<double>& spread, const std::string& tag) {
        std::vector<double> returns, positions_abs;
        double position = 0.0;
        double equity = 1.0;

        std::ofstream signal_out(std::string(QMP_RESULTS_DIR) + "/signal_" + tag + ".csv");
        signal_out << "date,spread,zscore,position\n";
        std::ofstream eq_out(std::string(QMP_RESULTS_DIR) + "/equity_" + tag + ".csv");
        eq_out << "date,return,equity\n";

        for (std::size_t t = train_window + zscore_window; t + 1 < T; ++t) {
            double mean = rolling_mean(spread, t, zscore_window);
            double sd = rolling_std(spread, t, zscore_window, mean);
            double z = sd > 0.0 ? (spread[t] - mean) / sd : 0.0;

            double target = position;
            if (position == 0.0) {
                if (z < -entry_z) target = 1.0;
                else if (z > entry_z) target = -1.0;
            } else if (position > 0.0) {
                target = (z > -exit_z) ? 0.0 : 1.0;
            } else {
                target = (z < exit_z) ? 0.0 : -1.0;
            }

            double spread_change = spread[t + 1] - spread[t];
            double cost = cost_one_way * std::abs(target - position);
            double pnl = position * spread_change - cost;

            returns.push_back(pnl);
            positions_abs.push_back(std::abs(target - position));
            equity *= (1.0 + pnl);
            position = target;

            signal_out << panel.dates[t] << "," << spread[t] << "," << z << "," << position << "\n";
            eq_out << panel.dates[t + 1] << "," << pnl << "," << equity << "\n";
        }

        return libquant_backtest::compute_metrics(returns, positions_abs);
    };

    auto metrics_kalman = run_backtest(spread_kalman, "kalman");
    auto metrics_static = run_backtest(spread_static, "static");

    std::ofstream metrics_out(std::string(QMP_RESULTS_DIR) + "/metrics.csv");
    metrics_out << "method,sharpe,max_drawdown,turnover,hit_rate,total_return,annualized_return\n";
    metrics_out << "kalman," << metrics_kalman.sharpe << "," << metrics_kalman.max_drawdown << ","
                << metrics_kalman.turnover << "," << metrics_kalman.hit_rate << "," << metrics_kalman.total_return
                << "," << metrics_kalman.annualized_return << "\n";
    metrics_out << "static," << metrics_static.sharpe << "," << metrics_static.max_drawdown << ","
                << metrics_static.turnover << "," << metrics_static.hit_rate << "," << metrics_static.total_return
                << "," << metrics_static.annualized_return << "\n";

    std::ofstream adf_out(std::string(QMP_RESULTS_DIR) + "/adf_test.csv");
    adf_out << "gamma,t_stat,critical_1pct,critical_5pct,critical_10pct,cointegrated_5pct\n";
    adf_out << adf.gamma << "," << adf.t_stat << "," << adf.critical_1pct << "," << adf.critical_5pct << ","
            << adf.critical_10pct << "," << (adf.cointegrated_5pct ? 1 : 0) << "\n";

    std::cout << "\n=== Kalman ===  Sharpe " << metrics_kalman.sharpe << "  MaxDD " << metrics_kalman.max_drawdown
              << "  AnnRet " << metrics_kalman.annualized_return << "\n";
    std::cout << "=== Static ===  Sharpe " << metrics_static.sharpe << "  MaxDD " << metrics_static.max_drawdown
              << "  AnnRet " << metrics_static.annualized_return << "\n";
    std::cout << "Wrote results to " << QMP_RESULTS_DIR << "\n";
    return 0;
}
