#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace libquant_backtest {

// Annualized Sharpe ratio from periodic (e.g. daily) returns:
// Sharpe = mean(r - rf) / stdev(r) * sqrt(periods_per_year).
// Uses the sample (n-1) standard deviation, the standard finance convention.
inline double sharpe_ratio(const std::vector<double>& returns, double risk_free_per_period = 0.0,
                            double periods_per_year = 252.0) {
    if (returns.size() < 2) return 0.0;
    double mean = 0.0;
    for (double r : returns) mean += (r - risk_free_per_period);
    mean /= static_cast<double>(returns.size());
    double var = 0.0;
    for (double r : returns) {
        double d = (r - risk_free_per_period) - mean;
        var += d * d;
    }
    var /= static_cast<double>(returns.size() - 1);
    double sd = std::sqrt(var);
    if (sd == 0.0) return 0.0;
    return (mean / sd) * std::sqrt(periods_per_year);
}

// Cumulative equity curve from periodic returns, starting at `start`.
inline std::vector<double> equity_curve(const std::vector<double>& returns, double start = 1.0) {
    std::vector<double> curve;
    curve.reserve(returns.size() + 1);
    double v = start;
    curve.push_back(v);
    for (double r : returns) {
        v *= (1.0 + r);
        curve.push_back(v);
    }
    return curve;
}

// Maximum peak-to-trough drawdown of an equity curve, as a positive
// fraction (0.25 == a 25% decline from the running peak).
inline double max_drawdown(const std::vector<double>& curve) {
    if (curve.empty()) return 0.0;
    double peak = curve[0];
    double worst = 0.0;
    for (double v : curve) {
        peak = std::max(peak, v);
        if (peak > 0.0) worst = std::max(worst, (peak - v) / peak);
    }
    return worst;
}

// Mean absolute period-over-period change in position size: a simple
// turnover proxy for accounting for transaction costs.
inline double turnover(const std::vector<double>& positions) {
    if (positions.size() < 2) return 0.0;
    double total = 0.0;
    for (std::size_t i = 1; i < positions.size(); ++i) total += std::abs(positions[i] - positions[i - 1]);
    return total / static_cast<double>(positions.size() - 1);
}

// Fraction of periods with strictly positive return.
inline double hit_rate(const std::vector<double>& returns) {
    if (returns.empty()) return 0.0;
    std::size_t wins = 0;
    for (double r : returns)
        if (r > 0.0) ++wins;
    return static_cast<double>(wins) / static_cast<double>(returns.size());
}

// Bundle of the metrics every project reports, computed with identical
// formulas/conventions so cross-project comparison (comparison/) is
// apples-to-apples.
struct BacktestMetrics {
    double sharpe = 0.0;
    double max_drawdown = 0.0;
    double turnover = 0.0;
    double hit_rate = 0.0;
    double total_return = 0.0;
    double annualized_return = 0.0;
};

inline BacktestMetrics compute_metrics(const std::vector<double>& returns, const std::vector<double>& positions,
                                        double periods_per_year = 252.0) {
    BacktestMetrics m;
    m.sharpe = sharpe_ratio(returns, 0.0, periods_per_year);
    auto curve = equity_curve(returns);
    m.max_drawdown = max_drawdown(curve);
    m.turnover = turnover(positions);
    m.hit_rate = hit_rate(returns);
    m.total_return = curve.empty() ? 0.0 : curve.back() - 1.0;
    double years = static_cast<double>(returns.size()) / periods_per_year;
    double final_value = curve.empty() ? 1.0 : curve.back();
    m.annualized_return = (years > 0.0 && final_value > 0.0) ? std::pow(final_value, 1.0 / years) - 1.0 : 0.0;
    return m;
}

}  // namespace libquant_backtest
