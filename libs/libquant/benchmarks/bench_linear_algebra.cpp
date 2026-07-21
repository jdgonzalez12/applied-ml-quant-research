// Chrono-based latency benchmark: libquant's from-scratch kernels vs Eigen,
// the reference oracle. Writes a CSV so the portfolio README can plot
// measured GFLOPS/latency without hand-transcribing numbers.
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "libquant/cholesky.hpp"
#include "libquant/lu.hpp"
#include "libquant/matrix.hpp"

using Clock = std::chrono::steady_clock;

namespace {

template <typename F>
double time_ns_median(F&& f, int repeats) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        auto t0 = Clock::now();
        f();
        auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::ofstream out("results.csv");
    out << "kernel,n,library,latency_ns\n";

    for (std::size_t n : {16u, 32u, 64u, 128u, 256u}) {
        Eigen::MatrixXd Ae = Eigen::MatrixXd::Random(static_cast<int>(n), static_cast<int>(n));
        Ae = Ae * Ae.transpose() + static_cast<double>(n) * Eigen::MatrixXd::Identity(static_cast<int>(n), static_cast<int>(n));
        Eigen::VectorXd be = Eigen::VectorXd::Random(static_cast<int>(n));

        libquant::Matrix<double> A(n, n), b(n, 1);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) A(i, j) = Ae(static_cast<int>(i), static_cast<int>(j));
            b(i, 0) = be(static_cast<int>(i));
        }

        const int repeats = static_cast<int>(n) <= 64 ? 50 : 10;

        double lu_ns = time_ns_median([&] { libquant::LU<double> lu(A); volatile auto x = lu.solve(b); (void)x; }, repeats);
        double eigen_lu_ns = time_ns_median([&] { volatile auto x = Ae.partialPivLu().solve(be).eval(); (void)x; }, repeats);

        double chol_ns = time_ns_median([&] { libquant::Cholesky<double> ch(A); volatile auto x = ch.solve(b); (void)x; }, repeats);
        double eigen_chol_ns = time_ns_median([&] { volatile auto x = Ae.llt().solve(be).eval(); (void)x; }, repeats);

        out << "lu_solve," << n << ",libquant," << lu_ns << "\n";
        out << "lu_solve," << n << ",eigen," << eigen_lu_ns << "\n";
        out << "cholesky_solve," << n << ",libquant," << chol_ns << "\n";
        out << "cholesky_solve," << n << ",eigen," << eigen_chol_ns << "\n";

        std::printf("n=%3zu  LU: libquant=%10.0f ns  eigen=%10.0f ns   Cholesky: libquant=%10.0f ns  eigen=%10.0f ns\n",
                     n, lu_ns, eigen_lu_ns, chol_ns, eigen_chol_ns);
    }

    std::printf("Wrote results.csv\n");
    return 0;
}
