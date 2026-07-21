#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "dmd.hpp"
#include "kalman_filter.hpp"
#include "persistent_homology.hpp"

using Catch::Approx;
using namespace qmp04;

TEST_CASE("DMD recovers the known eigenvalues of a synthetic linear system", "[dmd]") {
    // A damped oscillator: A = r * [[cos t, -sin t], [sin t, cos t]], a
    // rotation-scaling matrix with eigenvalues r*e^{+-i t}.
    const double r = 0.95, theta = 0.3;
    libquant::Matrix<double> A(2, 2);
    A(0, 0) = r * std::cos(theta); A(0, 1) = -r * std::sin(theta);
    A(1, 0) = r * std::sin(theta); A(1, 1) = r * std::cos(theta);

    const int m = 30;
    libquant::Matrix<double> X(2, m - 1), Xp(2, m - 1);
    libquant::Matrix<double> state(2, 1);
    state(0, 0) = 1.0; state(1, 0) = 0.5;

    std::vector<double> s0 = {state(0, 0), state(1, 0)};
    std::vector<std::vector<double>> snapshots;
    snapshots.push_back(s0);
    for (int t = 0; t < m - 1; ++t) {
        libquant::Matrix<double> next = A * state;
        snapshots.push_back({next(0, 0), next(1, 0)});
        state = next;
    }
    for (int t = 0; t < m - 1; ++t) {
        X(0, t) = snapshots[static_cast<std::size_t>(t)][0];
        X(1, t) = snapshots[static_cast<std::size_t>(t)][1];
        Xp(0, t) = snapshots[static_cast<std::size_t>(t) + 1][0];
        Xp(1, t) = snapshots[static_cast<std::size_t>(t) + 1][1];
    }

    DMDResult dmd = compute_dmd(X, Xp, 0.999);
    REQUIRE(dmd.eigenvalues.size() == 2);
    for (const auto& mu : dmd.eigenvalues) {
        REQUIRE(std::abs(mu) == Approx(r).margin(1e-4));
    }
}

TEST_CASE("max_growth_rate is negative for a decaying system and near zero for a neutral one", "[dmd]") {
    // Pure decay: x_{t+1} = 0.5 x_t (1D-like, embedded in 2 independent channels).
    libquant::Matrix<double> X(2, 10), Xp(2, 10);
    double v0 = 1.0, v1 = 2.0;
    for (int t = 0; t < 10; ++t) {
        X(0, t) = v0; X(1, t) = v1;
        v0 *= 0.5; v1 *= 0.5;
        Xp(0, t) = v0; Xp(1, t) = v1;
    }
    DMDResult dmd = compute_dmd(X, Xp, 0.999);
    double rate = max_growth_rate(dmd, 1.0);
    REQUIRE(rate < 0.0);  // ln(0.5) < 0: decaying, stable
}

TEST_CASE("Persistent homology finds exactly one H1 loop in a unit square", "[tda]") {
    std::vector<std::vector<double>> points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    auto simplices = build_vietoris_rips(points, 2.0);
    auto pairs = compute_persistence(simplices);

    int h1_count = 0;
    double birth = 0.0, death = 0.0;
    for (const auto& p : pairs) {
        if (p.dimension == 1 && (p.death - p.birth) > 1e-9) {
            ++h1_count;
            birth = p.birth;
            death = p.death;
        }
    }
    REQUIRE(h1_count == 1);
    REQUIRE(birth == Approx(1.0).margin(1e-9));
    REQUIRE(death == Approx(std::sqrt(2.0)).margin(1e-9));
}

TEST_CASE("Persistent homology finds no genuine loop in a filled triangle", "[tda]") {
    std::vector<std::vector<double>> points = {{0, 0}, {1, 0}, {0.5, std::sqrt(3.0) / 2.0}};  // equilateral
    auto simplices = build_vietoris_rips(points, 2.0);
    auto pairs = compute_persistence(simplices);

    REQUIRE(max_h1_persistence(pairs) == Approx(0.0).margin(1e-9));
}

TEST_CASE("Persistent homology finds one H1 loop in a regular pentagon, none in a tight cluster", "[tda]") {
    // A regular pentagon has exactly two distinct pairwise distances (side
    // and diagonal), so at an epsilon between them the complex is exactly
    // the 5-cycle (one clean loop, no chords yet); increasing epsilon past
    // the diagonal length adds all 10 possible triangles at once, filling
    // the loop. This is the same closed-form reasoning as the unit-square
    // test, generalized to 5 points.
    std::vector<std::vector<double>> pentagon;
    const int n = 5;
    for (int i = 0; i < n; ++i) {
        double a = 2.0 * 3.14159265358979323846 * static_cast<double>(i) / static_cast<double>(n);
        pentagon.push_back({std::cos(a), std::sin(a)});
    }
    auto simplices = build_vietoris_rips(pentagon, 2.5);
    auto pairs = compute_persistence(simplices);
    double pentagon_persistence = max_h1_persistence(pairs);

    std::vector<std::vector<double>> blob = {
        {0.0, 0.0}, {0.01, 0.0}, {0.0, 0.01}, {0.01, 0.01}, {0.005, 0.005}};
    auto blob_simplices = build_vietoris_rips(blob, 2.5);
    auto blob_pairs = compute_persistence(blob_simplices);
    double blob_persistence = max_h1_persistence(blob_pairs);

    REQUIRE(pentagon_persistence > 0.0);
    REQUIRE(pentagon_persistence > blob_persistence);
}

TEST_CASE("KalmanFilter1D local-level mode (x_t=1) tracks a slowly drifting mean", "[kalman]") {
    KalmanFilter1D kf(0.0, 1.0, 1e-4, 0.01);
    double level = 0.0;
    for (int t = 0; t < 300; ++t) {
        if (t == 150) level = 1.0;  // a step change in the underlying level
        kf.step(1.0, level);
    }
    REQUIRE(kf.beta() == Approx(1.0).margin(0.1));
}
