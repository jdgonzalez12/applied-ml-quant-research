#pragma once

#include <vector>

namespace qmp03 {

// Scalar discrete-time finite-horizon LQR via backward Riccati recursion
// (Brunton & Kutz, "Optimal full-state control: the linear quadratic
// regulator (LQR)"), specialized to the Almgren-Chriss optimal-execution
// problem: liquidate an inventory x_0 over T periods.
//
// State/control model: x_{k+1} = A x_k + B u_k, with A = 1, B = -1 (selling
// u_k shares reduces inventory by u_k each period). Per-period cost
// Q x_k^2 + R u_k^2 trades off holding risk (Q: proportional to price
// variance -- larger inventory held longer is riskier) against temporary
// market-impact cost (R: cost per share traded, roughly quadratic in trade
// size per Almgren-Chriss). A large terminal penalty Q_f forces x_T -> 0,
// approximating the hard constraint "fully liquidated by the deadline."
//
// The Bellman equation for this finite-horizon LQ problem is solved
// backward: P_T = Q_f, then for k = T-1 down to 0,
//   K_k = (R + B P_{k+1} B) ^ -1 * B P_{k+1} A
//   P_k = Q + A P_{k+1} A - (A P_{k+1} B)(R + B P_{k+1} B)^-1(B P_{k+1} A)
// giving the optimal feedback law u_k = -K_k x_k. For A=1, B=-1 this
// reduces to u_k = x_k * P_{k+1} / (R + P_{k+1}): sell a state-dependent
// fraction of the remaining inventory each period -- the discrete
// Almgren-Chriss trading trajectory.
struct LQRExecutionResult {
    std::vector<double> gains;      // K_k, k = 0..T-1
    std::vector<double> inventory;  // x_k, k = 0..T (x_0 given, x_T ~ 0)
    std::vector<double> trades;     // u_k, k = 0..T-1
    double total_cost = 0.0;        // sum_k (Q x_k^2 + R u_k^2)
};

inline LQRExecutionResult solve_lqr_execution(double x0, int T, double Q, double R, double Qf) {
    LQRExecutionResult res;
    std::vector<double> P(static_cast<std::size_t>(T) + 1, 0.0);
    P[static_cast<std::size_t>(T)] = Qf;

    res.gains.resize(static_cast<std::size_t>(T));
    for (int k = T - 1; k >= 0; --k) {
        double Pnext = P[static_cast<std::size_t>(k) + 1];
        double denom = R + Pnext;  // B^T P B = P for B=-1
        double K = Pnext / denom;  // u_k = -K_k x_k with the sign folded in below
        res.gains[static_cast<std::size_t>(k)] = K;
        // Mathematically P_k = Q + Pnext - Pnext^2/denom, but that subtracts
        // two nearly-equal O(Pnext) terms whenever Qf >> R (a realistic
        // regime here, since the terminal penalty is deliberately large) --
        // catastrophic cancellation that can lose all precision in the
        // O(R)-sized result. The algebraically identical Q + Pnext*R/denom
        // is a product/quotient instead of a subtraction of large terms,
        // and is numerically stable.
        P[static_cast<std::size_t>(k)] = Q + (Pnext * R) / denom;
    }

    res.inventory.resize(static_cast<std::size_t>(T) + 1);
    res.trades.resize(static_cast<std::size_t>(T));
    res.inventory[0] = x0;
    double cost = 0.0;
    for (int k = 0; k < T; ++k) {
        double x = res.inventory[static_cast<std::size_t>(k)];
        double u = res.gains[static_cast<std::size_t>(k)] * x;  // shares sold this period
        res.trades[static_cast<std::size_t>(k)] = u;
        res.inventory[static_cast<std::size_t>(k) + 1] = x - u;
        cost += Q * x * x + R * u * u;
    }
    res.total_cost = cost;
    return res;
}

// TWAP baseline: liquidate x0/T shares every period, ignoring the state
// entirely -- the naive execution schedule LQR is compared against.
inline LQRExecutionResult solve_twap_execution(double x0, int T, double Q, double R) {
    LQRExecutionResult res;
    res.inventory.resize(static_cast<std::size_t>(T) + 1);
    res.trades.assign(static_cast<std::size_t>(T), x0 / static_cast<double>(T));
    res.inventory[0] = x0;
    double cost = 0.0;
    for (int k = 0; k < T; ++k) {
        double x = res.inventory[static_cast<std::size_t>(k)];
        double u = res.trades[static_cast<std::size_t>(k)];
        res.inventory[static_cast<std::size_t>(k) + 1] = x - u;
        cost += Q * x * x + R * u * u;
    }
    res.total_cost = cost;
    return res;
}

}  // namespace qmp03
