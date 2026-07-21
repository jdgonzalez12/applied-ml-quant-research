#pragma once

#include <algorithm>
#include <cmath>

namespace qmp04 {

// Identical implementation to Project 3's KalmanFilter1D (see
// projects/03-kalman-pairs-control/include/kalman_filter.hpp for the full
// derivation) -- reused here as Project 4's third regime-detection
// baseline. Called with x_t == 1 always, the observation model
// y_t = beta_t * 1 + v_t reduces exactly to a local-level tracker
// level_t = level_{t-1} + w_t, y_t = level_t + v_t, and the standardized
// innovation e_t/sqrt(S_t) becomes a CUSUM-style break-detection statistic:
// a return that is many standard deviations away from the filter's current
// tracked level is itself a signal of a regime change.
class KalmanFilter1D {
public:
    KalmanFilter1D(double initial_beta, double initial_variance, double process_var, double obs_var)
        : beta_(initial_beta), P_(initial_variance), Q_(process_var), R_(obs_var) {}

    struct StepResult {
        double beta = 0.0;
        double innovation = 0.0;
        double innovation_var = 0.0;
        double standardized_innovation = 0.0;
    };

    StepResult step(double x_t, double y_t) {
        const double beta_pred = beta_;
        const double P_pred = P_ + Q_;

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

}  // namespace qmp04
