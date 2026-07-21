#pragma once

#include <algorithm>
#include <cmath>

namespace qmp03 {

// Scalar Kalman filter for a time-varying hedge ratio beta_t in the
// observation model y_t = beta_t x_t + v_t, v_t ~ N(0, R), with beta
// following a random walk beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
// (Brunton & Kutz, "Optimal full-state estimation: The Kalman filter").
//
// The update is the minimum-variance (BLUE) recursive combination of the
// prior prediction and the new observation, weighted inversely by their
// respective variances -- the Kalman gain K_t is exactly that optimal
// weight, derived by minimizing the posterior variance P_t over K:
//   P_t(K) = (1 - K x_t)^2 P_pred + K^2 R  =>  dP/dK = 0  =>  K = P_pred x_t / S_t.
class KalmanFilter1D {
public:
    KalmanFilter1D(double initial_beta, double initial_variance, double process_var, double obs_var)
        : beta_(initial_beta), P_(initial_variance), Q_(process_var), R_(obs_var) {}

    struct StepResult {
        double beta = 0.0;                    // posterior hedge-ratio estimate
        double innovation = 0.0;              // e_t = y_t - x_t * beta_predicted
        double innovation_var = 0.0;          // S_t = x_t^2 P_predicted + R
        double standardized_innovation = 0.0; // e_t / sqrt(S_t): reused as a break-detection statistic in Project 4
    };

    StepResult step(double x_t, double y_t) {
        // Predict.
        const double beta_pred = beta_;  // random-walk model: E[beta_t | t-1] = beta_{t-1}
        const double P_pred = P_ + Q_;

        // Update.
        const double e = y_t - x_t * beta_pred;
        const double S = x_t * x_t * P_pred + R_;
        const double K = (S > 0.0) ? (P_pred * x_t / S) : 0.0;

        beta_ = beta_pred + K * e;
        P_ = (1.0 - K * x_t) * P_pred;

        StepResult r;
        r.beta = beta_;
        r.innovation = e;
        r.innovation_var = S;
        r.standardized_innovation = (S > 0.0) ? e / std::sqrt(S) : 0.0;
        return r;
    }

    [[nodiscard]] double beta() const noexcept { return beta_; }
    [[nodiscard]] double variance() const noexcept { return P_; }

private:
    double beta_;
    double P_;
    double Q_;
    double R_;
};

}  // namespace qmp03
