#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "libquant/general_eigen.hpp"
#include "libquant/matrix.hpp"
#include "libquant/svd.hpp"

namespace qmp04 {

struct DMDResult {
    std::vector<std::complex<double>> eigenvalues;  // discrete-time DMD eigenvalues mu_i
    std::size_t rank = 0;                            // truncation rank actually used
};

// Exact DMD (Tu, Rowley, Luchtenburg, Brunton, Kutz 2014; Brunton & Kutz,
// "Dynamic Mode Decomposition"), eigenvalues only. Given snapshot matrices
// X = [x_1 ... x_{m-1}], X' = [x_2 ... x_m] (each column a system state):
//   1. X = U Sigma V^T (libquant::SVD), truncated to rank r by an energy
//      (cumulative singular-value) threshold.
//   2. Reduced operator: A_tilde = U_r^T X' V_r Sigma_r^{-1}  (r x r) --
//      the best-fit linear operator advancing the state one step, in the
//      SVD-reduced coordinate basis.
//   3. Eigenvalues mu_i of A_tilde (libquant::GeneralEigen) are the DMD
//      eigenvalues; under the Koopman-operator interpretation, they
//      approximate the leading eigenvalues of the (generally infinite-
//      dimensional) Koopman operator advancing observables of the true
//      nonlinear dynamics. The continuous-time growth rate is
//      omega_i = ln(mu_i) / dt: Re(omega_i) > 0 signals an unstable
//      (expanding) mode -- the regime-break signal this project tracks.
inline DMDResult compute_dmd(const libquant::Matrix<double>& X, const libquant::Matrix<double>& Xp,
                              double energy_threshold = 0.99) {
    DMDResult result;

    // libquant::SVD requires rows >= cols. DMD snapshot matrices are
    // typically wide (state dimension << number of snapshots), so when X
    // is wide, decompose X^T instead: if X = U Sigma V^T then
    // X^T = V Sigma U^T is exactly the SVD of X^T, so swapping the roles of
    // U and V recovers X's own factors.
    const bool wide = X.rows() < X.cols();
    libquant::SVD<double> svd(wide ? X.transpose() : X);
    const auto& sv = svd.singular_values();
    const libquant::Matrix<double>& U = wide ? svd.V() : svd.U();
    const libquant::Matrix<double>& V = wide ? svd.U() : svd.V();

    double total = 0.0;
    for (double s : sv) total += s;
    std::size_t r = 1;
    if (total > 0.0) {
        double cum = 0.0;
        for (std::size_t i = 0; i < sv.size(); ++i) {
            cum += sv[i];
            r = i + 1;
            if (cum / total >= energy_threshold) break;
        }
    }
    result.rank = r;

    const std::size_t m = X.rows();
    const std::size_t n = X.cols();
    libquant::Matrix<double> Ur(m, r), Vr(n, r);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < r; ++k) Ur(i, k) = U(i, k);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < r; ++k) Vr(i, k) = V(i, k);

    libquant::Matrix<double> Sinv_r(r, r, 0.0);
    for (std::size_t k = 0; k < r; ++k) Sinv_r(k, k) = (sv[k] > 1e-12) ? 1.0 / sv[k] : 0.0;

    // A_tilde = Ur^T Xp Vr Sinv_r
    libquant::Matrix<double> Atilde = Ur.transpose() * Xp * Vr * Sinv_r;

    libquant::GeneralEigen<double> eig(Atilde);
    result.eigenvalues = eig.eigenvalues();
    return result;
}

// Continuous-time growth rate of the least stable (fastest-growing) DMD
// mode: max_i Re(ln(mu_i)/dt). A positive value indicates locally unstable
// (expanding) dynamics.
inline double max_growth_rate(const DMDResult& dmd, double dt) {
    double best = -1e18;
    for (const auto& mu : dmd.eigenvalues) {
        double mag = std::abs(mu);
        if (mag <= 0.0) continue;
        std::complex<double> omega = std::log(mu) / dt;
        best = std::max(best, omega.real());
    }
    return best;
}

}  // namespace qmp04
