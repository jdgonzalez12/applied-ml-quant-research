// Project 2: RMT/PCA statistical arbitrage.
//
// Pipeline (see README.md for the full mathematical derivation):
//   1. Trailing 60-day correlation matrix of 9 SPDR sector ETFs.
//   2. Marchenko-Pastur eigenvalue denoising -> number of genuine factors.
//   3. Eigenportfolio factor returns (Avellaneda-Lee weighting).
//   4. Per-asset regression on the factors -> idiosyncratic residual.
//   5. OU fit of the cumulative residual -> s-score.
//   6. Hysteresis entry/exit trading rule, walk-forward, one day at a time.
//
// All numerical work is C++; this program only writes CSV result artifacts
// for the Python plotting layer.
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

#include "csv_loader.hpp"
#include "marchenko_pastur.hpp"
#include "ou_signal.hpp"

#ifndef QMP_DATA_DIR
#define QMP_DATA_DIR "data/raw/equities"
#endif
#ifndef QMP_RESULTS_DIR
#define QMP_RESULTS_DIR "results"
#endif

int main() {
    using namespace qmp02;

    const std::vector<std::string> symbols = {"xlf", "xlk", "xle", "xlv", "xly", "xlp", "xli", "xlu", "xlb"};
    const std::size_t N = symbols.size();

    std::cout << "Loading price panel for " << N << " sector ETFs...\n";
    PricePanel panel = load_price_panel(QMP_DATA_DIR, symbols);
    libquant::Matrix<double> R = log_returns(panel.prices);  // (T-1) x N; row i realizes on panel.dates[i+1]
    const std::size_t T = R.rows();
    std::cout << "  " << panel.dates.size() << " aligned trading days, " << T << " return observations.\n";

    // Benchmark: SPY, same alignment.
    auto spy_prices = load_adjclose(std::string(QMP_DATA_DIR) + "/spy.csv");

    const std::size_t lookback = 60;
    const double entry_threshold = 1.25;
    const double exit_threshold = 0.5;
    const double min_half_life = 1.0;
    const double max_half_life = 30.0;
    const double cost_one_way = 5e-4;  // 5 bps per unit position change

    std::filesystem::create_directories(QMP_RESULTS_DIR);
    std::ofstream spectrum_out(std::string(QMP_RESULTS_DIR) + "/eigen_spectrum.csv");
    spectrum_out << "window_end_date,eigenvalue_rank,eigenvalue,lambda_plus,lambda_minus\n";
    std::ofstream loadings_out(std::string(QMP_RESULTS_DIR) + "/factor_loadings.csv");
    loadings_out << "window_end_date,symbol,pc,loading\n";
    std::ofstream sscore_out(std::string(QMP_RESULTS_DIR) + "/sscore.csv");
    sscore_out << "date,symbol,s_score,half_life,num_factors,position\n";
    std::ofstream equity_out(std::string(QMP_RESULTS_DIR) + "/equity_curve.csv");
    equity_out << "date,strategy_return,strategy_equity,spy_equity\n";

    std::vector<double> position(N, 0.0);
    std::vector<double> portfolio_returns;
    std::vector<double> mean_abs_position;
    portfolio_returns.reserve(T);

    double strategy_equity = 1.0;
    double spy_equity = 1.0;
    double spy_prev = -1.0;

    for (std::size_t end = lookback; end + 1 < T; ++end) {
        libquant::Matrix<double> Rw(lookback, N);
        for (std::size_t t = 0; t < lookback; ++t)
            for (std::size_t j = 0; j < N; ++j) Rw(t, j) = R(end - lookback + t, j);

        libquant::Matrix<double> C = correlation_matrix(Rw);
        double q = static_cast<double>(N) / static_cast<double>(lookback);
        MPResult mp = marchenko_pastur_filter(C, q);

        std::size_t num_factors = 0;
        for (double lam : mp.eigenvalues)
            if (lam > mp.lambda_plus) ++num_factors;
        num_factors = std::clamp<std::size_t>(num_factors, 1, 3);

        std::vector<double> sigma(N, 0.0);
        for (std::size_t j = 0; j < N; ++j) {
            double mean = 0.0;
            for (std::size_t t = 0; t < lookback; ++t) mean += Rw(t, j);
            mean /= static_cast<double>(lookback);
            double var = 0.0;
            for (std::size_t t = 0; t < lookback; ++t) {
                double d = Rw(t, j) - mean;
                var += d * d;
            }
            var /= static_cast<double>(lookback - 1);
            sigma[j] = std::sqrt(var);
        }

        // Eigenportfolio (Avellaneda-Lee) weights: w_j = V(j,k)/sigma_j,
        // normalized to unit L2 norm, applied to raw returns to build each
        // factor's return series.
        std::vector<std::vector<double>> weights(num_factors, std::vector<double>(N));
        libquant::Matrix<double> F(lookback, num_factors);
        for (std::size_t k = 0; k < num_factors; ++k) {
            double norm = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                double w = sigma[j] > 0.0 ? mp.eigenvectors(j, k) / sigma[j] : 0.0;
                weights[k][j] = w;
                norm += w * w;
            }
            norm = std::sqrt(norm);
            if (norm == 0.0) norm = 1.0;
            for (std::size_t j = 0; j < N; ++j) weights[k][j] /= norm;
            for (std::size_t t = 0; t < lookback; ++t) {
                double f = 0.0;
                for (std::size_t j = 0; j < N; ++j) f += weights[k][j] * Rw(t, j);
                F(t, k) = f;
            }
        }

        const std::string& today_date = panel.dates[end + 1];

        if ((end - lookback) % 63 == 0) {  // quarterly snapshots keep the plot files small
            for (std::size_t r = 0; r < mp.eigenvalues.size(); ++r)
                spectrum_out << today_date << "," << (r + 1) << "," << mp.eigenvalues[r] << "," << mp.lambda_plus
                             << "," << mp.lambda_minus << "\n";
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t k = 0; k < std::min<std::size_t>(3, N); ++k)
                    loadings_out << today_date << "," << symbols[j] << "," << (k + 1) << "," << mp.eigenvectors(j, k)
                                 << "\n";
        }

        libquant::Matrix<double> design(lookback, num_factors + 1);
        for (std::size_t t = 0; t < lookback; ++t) {
            design(t, 0) = 1.0;
            for (std::size_t k = 0; k < num_factors; ++k) design(t, k + 1) = F(t, k);
        }

        std::vector<double> day_pnl(N, 0.0);
        double turnover_sum = 0.0;

        for (std::size_t j = 0; j < N; ++j) {
            libquant::Matrix<double> yj(lookback, 1);
            for (std::size_t t = 0; t < lookback; ++t) yj(t, 0) = Rw(t, j);
            libquant::Matrix<double> beta = libquant::ols(design, yj);

            std::vector<double> X(lookback);
            double cum = 0.0;
            for (std::size_t t = 0; t < lookback; ++t) {
                double pred = beta(0, 0);
                for (std::size_t k = 0; k < num_factors; ++k) pred += beta(k + 1, 0) * F(t, k);
                cum += (Rw(t, j) - pred);
                X[t] = cum;
            }

            OUFit fit = fit_ou(X);
            double score = s_score(fit, X.back());
            bool tradable = fit.valid && fit.half_life >= min_half_life && fit.half_life <= max_half_life;

            double target = position[j];
            if (!tradable) {
                target = 0.0;
            } else if (position[j] == 0.0) {
                if (score < -entry_threshold) target = 1.0;
                else if (score > entry_threshold) target = -1.0;
                else target = 0.0;
            } else if (position[j] > 0.0) {
                target = (score > -exit_threshold) ? 0.0 : 1.0;
            } else {
                target = (score < exit_threshold) ? 0.0 : -1.0;
            }

            std::vector<double> f_next(num_factors, 0.0);
            for (std::size_t k = 0; k < num_factors; ++k) {
                double f = 0.0;
                for (std::size_t jj = 0; jj < N; ++jj) f += weights[k][jj] * R(end, jj);
                f_next[k] = f;
            }
            double pred_next = beta(0, 0);
            for (std::size_t k = 0; k < num_factors; ++k) pred_next += beta(k + 1, 0) * f_next[k];
            double resid_next = R(end, j) - pred_next;

            double trade_cost = cost_one_way * std::abs(target - position[j]);
            day_pnl[j] = position[j] * resid_next - trade_cost;
            turnover_sum += std::abs(target - position[j]);

            sscore_out << today_date << "," << symbols[j] << "," << score << "," << fit.half_life << ","
                       << num_factors << "," << target << "\n";

            position[j] = target;
        }

        double day_return = 0.0;
        for (double p : day_pnl) day_return += p;
        day_return /= static_cast<double>(N);
        portfolio_returns.push_back(day_return);
        mean_abs_position.push_back(turnover_sum / static_cast<double>(N));

        strategy_equity *= (1.0 + day_return);
        double spy_now = spy_prices.count(today_date) ? spy_prices.at(today_date) : -1.0;
        if (spy_prev > 0.0 && spy_now > 0.0) spy_equity *= (spy_now / spy_prev);
        if (spy_now > 0.0) spy_prev = spy_now;

        equity_out << today_date << "," << day_return << "," << strategy_equity << "," << spy_equity << "\n";
    }

    auto metrics = libquant_backtest::compute_metrics(portfolio_returns, mean_abs_position);
    std::ofstream metrics_out(std::string(QMP_RESULTS_DIR) + "/metrics.csv");
    metrics_out << "metric,value\n";
    metrics_out << "sharpe," << metrics.sharpe << "\n";
    metrics_out << "max_drawdown," << metrics.max_drawdown << "\n";
    metrics_out << "turnover," << metrics.turnover << "\n";
    metrics_out << "hit_rate," << metrics.hit_rate << "\n";
    metrics_out << "total_return," << metrics.total_return << "\n";
    metrics_out << "annualized_return," << metrics.annualized_return << "\n";
    metrics_out << "n_days," << portfolio_returns.size() << "\n";

    std::cout << "Sharpe:            " << metrics.sharpe << "\n";
    std::cout << "Max drawdown:      " << metrics.max_drawdown << "\n";
    std::cout << "Annualized return: " << metrics.annualized_return << "\n";
    std::cout << "Hit rate:          " << metrics.hit_rate << "\n";
    std::cout << "Wrote results to " << QMP_RESULTS_DIR << "\n";
    return 0;
}
