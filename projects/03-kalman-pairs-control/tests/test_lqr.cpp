#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "riccati_lqr.hpp"

using Catch::Approx;
using namespace qmp03;

TEST_CASE("LQR execution never costs more than TWAP under the same cost functional", "[lqr]") {
    // LQR is the argmin of exactly this cost functional over all feasible
    // trading schedules, so it must weakly dominate any fixed alternative
    // schedule, including TWAP -- a strong, simple correctness check.
    auto lqr = solve_lqr_execution(100000.0, 20, 1e-6, 1e-8, 1e6);
    auto twap = solve_twap_execution(100000.0, 20, 1e-6, 1e-8);
    REQUIRE(lqr.total_cost <= twap.total_cost);
}

TEST_CASE("LQR execution liquidates essentially the full inventory by the horizon", "[lqr]") {
    auto lqr = solve_lqr_execution(50000.0, 15, 1e-6, 1e-8, 1e6);
    REQUIRE(lqr.inventory.back() == Approx(0.0).margin(50.0));  // near-zero relative to a 50,000-share start
    for (std::size_t k = 1; k < lqr.inventory.size(); ++k)
        REQUIRE(lqr.inventory[k] <= lqr.inventory[k - 1] + 1e-9);  // monotonically liquidating
}

TEST_CASE("With zero holding-risk penalty, LQR reduces to uniform (TWAP-equivalent) slicing", "[lqr]") {
    // With Q=0, the only cost is impact R*u_k^2 subject to fully liquidating
    // by T; minimizing a sum of squares under a fixed-sum constraint is
    // minimized by the uniform split -- exactly TWAP's schedule.
    auto lqr = solve_lqr_execution(90000.0, 10, 0.0, 1e-8, 1e8);
    double expected_trade = 90000.0 / 10.0;
    for (double u : lqr.trades) REQUIRE(u == Approx(expected_trade).epsilon(1e-3));
}

TEST_CASE("Higher risk aversion front-loads execution more aggressively", "[lqr]") {
    auto low_risk = solve_lqr_execution(100000.0, 20, 1e-8, 1e-8, 1e6);
    auto high_risk = solve_lqr_execution(100000.0, 20, 1e-4, 1e-8, 1e6);
    // The first trade should be a larger fraction of inventory when holding
    // risk is penalized more heavily.
    REQUIRE(high_risk.trades[0] > low_risk.trades[0]);
}
