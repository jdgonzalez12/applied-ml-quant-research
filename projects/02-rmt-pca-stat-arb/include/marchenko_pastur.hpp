#pragma once

#include <cmath>
#include <vector>

#include "libquant/matrix.hpp"
#include "libquant/symmetric_eigen.hpp"

namespace qmp02 {

struct MPResult {
    double q = 0.0;             // N/T, the matrix aspect ratio
    double lambda_minus = 0.0;  // MP lower edge
    double lambda_plus = 0.0;   // MP upper edge
    std::vector<double> eigenvalues;           // descending, raw correlation-matrix eigenvalues
    std::vector<double> eigenvalues_denoised;  // trace-preserving MP-filtered eigenvalues
    libquant::Matrix<double> eigenvectors;     // N x N, column i is the eigenvector for eigenvalues[i]
};

// Sample correlation matrix of a T x N return window: each column is
// standardized to zero mean / unit variance, then C = Z^T Z / (T-1).
inline libquant::Matrix<double> correlation_matrix(const libquant::Matrix<double>& R) {
    const std::size_t T = R.rows(), N = R.cols();
    libquant::Matrix<double> Z(T, N);
    for (std::size_t j = 0; j < N; ++j) {
        double mean = 0.0;
        for (std::size_t t = 0; t < T; ++t) mean += R(t, j);
        mean /= static_cast<double>(T);
        double var = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            double d = R(t, j) - mean;
            var += d * d;
        }
        var /= static_cast<double>(T - 1);
        double sd = std::sqrt(var);
        for (std::size_t t = 0; t < T; ++t) Z(t, j) = sd > 0.0 ? (R(t, j) - mean) / sd : 0.0;
    }
    libquant::Matrix<double> C(N, N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (std::size_t t = 0; t < T; ++t) s += Z(t, i) * Z(t, j);
            C(i, j) = s / static_cast<double>(T - 1);
        }
    return C;
}

// Marchenko-Pastur eigenvalue denoising (Laloux, Cizeau, Bouchaud & Potters
// 1999; the classical random-matrix-theory treatment of sample correlation
// matrices). For an N x T matrix of i.i.d.-noise standardized returns with
// aspect ratio q = N/T, the eigenvalues of the sample correlation matrix
// fill the deterministic bulk [lambda_minus, lambda_plus] = (1 -/+
// sqrt(q))^2 as T,N -> infinity. Empirical eigenvalues below lambda_plus
// are therefore statistically indistinguishable from pure noise; eigenvalues
// above the edge indicate genuine common factors. Noise eigenvalues are
// replaced by their shared average, preserving the matrix trace (= N, since
// C is a correlation matrix) so the denoised matrix stays a valid
// correlation-like operator.
inline MPResult marchenko_pastur_filter(const libquant::Matrix<double>& C, double q) {
    libquant::SymmetricEigen<double> eig(C);
    MPResult res;
    res.q = q;
    res.lambda_minus = (1.0 - std::sqrt(q)) * (1.0 - std::sqrt(q));
    res.lambda_plus = (1.0 + std::sqrt(q)) * (1.0 + std::sqrt(q));
    res.eigenvalues = eig.eigenvalues();
    res.eigenvectors = eig.eigenvectors();

    double noise_sum = 0.0;
    std::size_t noise_count = 0;
    for (double lam : res.eigenvalues) {
        if (lam <= res.lambda_plus) {
            noise_sum += lam;
            ++noise_count;
        }
    }
    double noise_avg = noise_count > 0 ? noise_sum / static_cast<double>(noise_count) : 0.0;

    res.eigenvalues_denoised.resize(res.eigenvalues.size());
    for (std::size_t i = 0; i < res.eigenvalues.size(); ++i)
        res.eigenvalues_denoised[i] = (res.eigenvalues[i] > res.lambda_plus) ? res.eigenvalues[i] : noise_avg;
    return res;
}

}  // namespace qmp02
