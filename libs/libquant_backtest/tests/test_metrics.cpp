#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "libquant_backtest/metrics.hpp"
#include "libquant_backtest/rolling_window.hpp"
#include "libquant_backtest/walk_forward.hpp"

using Catch::Approx;
namespace bt = libquant_backtest;

TEST_CASE("equity_curve and max_drawdown on a known path", "[metrics]") {
    // +10%, -20%, +10%: curve is 1.0 -> 1.1 -> 0.88 -> 0.968.
    std::vector<double> returns = {0.10, -0.20, 0.10};
    auto curve = bt::equity_curve(returns);
    REQUIRE(curve.size() == 4);
    REQUIRE(curve[0] == Approx(1.0));
    REQUIRE(curve[1] == Approx(1.1));
    REQUIRE(curve[2] == Approx(0.88));
    REQUIRE(curve[3] == Approx(0.968));

    // Peak is 1.1, trough is 0.88: drawdown = (1.1-0.88)/1.1.
    REQUIRE(bt::max_drawdown(curve) == Approx((1.1 - 0.88) / 1.1).margin(1e-9));
}

TEST_CASE("sharpe_ratio is zero for a no-signal (all-zero) strategy", "[metrics]") {
    std::vector<double> returns(20, 0.0);
    REQUIRE(bt::sharpe_ratio(returns) == Approx(0.0));
}

TEST_CASE("sharpe_ratio is positive for a strictly positive, low-variance return stream", "[metrics]") {
    std::vector<double> returns = {0.001, 0.0012, 0.0009, 0.0011, 0.0010};
    REQUIRE(bt::sharpe_ratio(returns) > 0.0);
}

TEST_CASE("turnover and hit_rate on simple series", "[metrics]") {
    std::vector<double> positions = {0.0, 1.0, 1.0, -1.0};
    // |1-0| + |1-1| + |-1-1| = 1 + 0 + 2 = 3, over 3 steps -> mean 1.0.
    REQUIRE(bt::turnover(positions) == Approx(1.0));

    std::vector<double> returns = {0.01, -0.02, 0.03, -0.01};
    REQUIRE(bt::hit_rate(returns) == Approx(0.5));
}

TEST_CASE("compute_metrics no-signal strategy has zero turnover, sharpe, and only cost-driven drawdown", "[metrics]") {
    std::vector<double> returns(10, 0.0);
    std::vector<double> positions(10, 0.0);
    auto m = bt::compute_metrics(returns, positions);
    REQUIRE(m.sharpe == Approx(0.0));
    REQUIRE(m.turnover == Approx(0.0));
    REQUIRE(m.total_return == Approx(0.0));
    REQUIRE(m.max_drawdown == Approx(0.0));
}

TEST_CASE("walk_forward_splits never lets a test window precede its training data", "[walk_forward]") {
    auto splits = bt::walk_forward_splits(100, 50, 10, 10);
    REQUIRE_FALSE(splits.empty());
    for (const auto& s : splits) {
        REQUIRE(s.train_start == 0);
        REQUIRE(s.test_start == s.train_end);   // no gap, no overlap
        REQUIRE(s.test_start >= s.train_end);   // test strictly follows train: no look-ahead
        REQUIRE(s.test_end <= 100);
    }
    // Splits should tile forward monotonically.
    for (std::size_t i = 1; i < splits.size(); ++i)
        REQUIRE(splits[i].train_end > splits[i - 1].train_end);
}

TEST_CASE("rolling_window_run covers the expected number of windows", "[rolling_window]") {
    auto results = bt::rolling_window_run<std::size_t>(100, 20, 10, [](std::size_t start, std::size_t end) {
        return end - start;
    });
    // windows starting at 0,10,20,...,80 with window=20 fitting within 100: last start is 80 (80+20=100).
    REQUIRE(results.size() == 9);
    for (auto len : results) REQUIRE(len == 20);
}
