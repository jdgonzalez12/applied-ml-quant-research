// Project 4: regime-detection shootout -- DMD/Koopman spectral instability
// vs. TDA persistent homology vs. a Kalman-innovation monitor, applied to
// the identical instrument set and compared against hand-labeled crisis
// windows (2008 GFC, 2020 COVID).
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "libquant/matrix.hpp"

#include "csv_loader.hpp"
#include "dmd.hpp"
#include "kalman_filter.hpp"
#include "persistent_homology.hpp"

#ifndef QMP_DATA_DIR
#define QMP_DATA_DIR "data/raw/equities"
#endif
#ifndef QMP_MACRO_DIR
#define QMP_MACRO_DIR "data/raw/macro"
#endif
#ifndef QMP_RESULTS_DIR
#define QMP_RESULTS_DIR "results"
#endif

namespace {

struct CrisisWindow {
    std::string name, start, end;
};

bool in_window(const std::string& date, const CrisisWindow& w) { return date >= w.start && date <= w.end; }

double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}
double stdev_of(const std::vector<double>& v, double m) {
    double s = 0.0;
    for (double x : v) { double d = x - m; s += d * d; }
    return v.size() > 1 ? std::sqrt(s / static_cast<double>(v.size() - 1)) : 0.0;
}

}  // namespace

int main() {
    using namespace qmp04;

    const std::vector<std::string> sectors = {"xlf", "xlk", "xle", "xlv", "xly", "xlp", "xli", "xlu", "xlb"};
    std::cout << "Loading sector panel (for DMD) and SPY (for TDA/Kalman)...\n";
    PricePanel panel = load_price_panel(QMP_DATA_DIR, sectors);
    libquant::Matrix<double> R = log_returns(panel.prices);  // (T-1) x 9, row i realizes on panel.dates[i+1]
    const std::size_t T = R.rows();

    auto spy_close = load_adjclose(std::string(QMP_DATA_DIR) + "/spy.csv");
    std::vector<double> spy_ret(T, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
        const std::string &d0 = panel.dates[t], &d1 = panel.dates[t + 1];
        if (spy_close.count(d0) && spy_close.count(d1)) spy_ret[t] = std::log(spy_close.at(d1) / spy_close.at(d0));
    }

    const std::vector<CrisisWindow> crises = {
        {"2008 GFC", "2008-09-01", "2009-03-31"},
        {"2020 COVID", "2020-02-20", "2020-04-30"},
    };

    std::filesystem::create_directories(QMP_RESULTS_DIR);

    // --- DMD: rolling window over the 9-sector return panel. -------------
    const std::size_t dmd_window = 60, dmd_step = 5;
    std::map<std::size_t, double> dmd_signal;  // index into R (day t, using window ending at t) -> growth rate
    for (std::size_t end = dmd_window; end + 1 < T; end += dmd_step) {
        libquant::Matrix<double> X(sectors.size(), dmd_window - 1), Xp(sectors.size(), dmd_window - 1);
        for (std::size_t t = 0; t < dmd_window - 1; ++t)
            for (std::size_t j = 0; j < sectors.size(); ++j) {
                X(j, t) = R(end - dmd_window + t, j);
                Xp(j, t) = R(end - dmd_window + t + 1, j);
            }
        DMDResult dmd = compute_dmd(X, Xp, 0.99);
        dmd_signal[end] = max_growth_rate(dmd, 1.0);
    }
    std::cout << "  DMD: " << dmd_signal.size() << " windows computed.\n";

    // --- TDA: rolling window, Takens embedding of SPY log returns. -------
    const std::size_t tda_window = 60, tda_step = 5, embed_dim = 3, embed_delay = 1;
    std::map<std::size_t, double> tda_signal;
    for (std::size_t end = tda_window; end + 1 < T; end += tda_step) {
        std::vector<std::vector<double>> cloud;
        std::size_t start = end - tda_window;
        for (std::size_t i = start; i + (embed_dim - 1) * embed_delay < end; ++i) {
            std::vector<double> pt(embed_dim);
            for (std::size_t d = 0; d < embed_dim; ++d) pt[d] = spy_ret[i + d * embed_delay];
            cloud.push_back(pt);
        }
        // k-NN-derived epsilon: the 15th percentile of all pairwise distances,
        // keeping the complex sparse (Carlsson & Vejdemo-Johansson's
        // guidance for tractable Vietoris-Rips construction).
        std::vector<double> dists;
        for (std::size_t i = 0; i < cloud.size(); ++i)
            for (std::size_t j = i + 1; j < cloud.size(); ++j) {
                double s = 0.0;
                for (std::size_t d = 0; d < embed_dim; ++d) { double dd = cloud[i][d] - cloud[j][d]; s += dd * dd; }
                dists.push_back(std::sqrt(s));
            }
        if (dists.size() < 3) continue;
        std::sort(dists.begin(), dists.end());
        double epsilon = dists[dists.size() / 7];  // ~15th percentile
        double median_dist = dists[dists.size() / 2];

        auto simplices = build_vietoris_rips(cloud, epsilon);
        auto pairs = compute_persistence(simplices);
        double raw = max_h1_persistence(pairs);
        tda_signal[end] = median_dist > 0.0 ? raw / median_dist : 0.0;  // scale-normalized
    }
    std::cout << "  TDA: " << tda_signal.size() << " windows computed.\n";

    // --- Kalman-innovation monitor: daily, local-level model on SPY. -----
    std::vector<double> kalman_signal(T, 0.0);
    {
        double init_mean = mean_of(std::vector<double>(spy_ret.begin(), spy_ret.begin() + static_cast<long>(dmd_window)));
        double init_var = stdev_of(std::vector<double>(spy_ret.begin(), spy_ret.begin() + static_cast<long>(dmd_window)), init_mean);
        KalmanFilter1D kf(init_mean, 1e-4, 1e-7, init_var * init_var);
        for (std::size_t t = 0; t < T; ++t) kalman_signal[t] = std::abs(kf.step(1.0, spy_ret[t]).standardized_innovation);
    }

    // --- Write combined signal table + crisis labels. ---------------------
    std::ofstream signals_out(std::string(QMP_RESULTS_DIR) + "/regime_signals.csv");
    signals_out << "date,dmd_growth_rate,tda_persistence,kalman_innovation,crisis_label\n";
    double last_dmd = 0.0, last_tda = 0.0;
    for (std::size_t t = dmd_window; t + 1 < T; ++t) {
        if (dmd_signal.count(t)) last_dmd = dmd_signal[t];
        if (tda_signal.count(t)) last_tda = tda_signal[t];
        const std::string& date = panel.dates[t + 1];
        int label = 0;
        for (const auto& c : crises)
            if (in_window(date, c)) label = 1;
        signals_out << date << "," << last_dmd << "," << last_tda << "," << kalman_signal[t] << "," << label << "\n";
    }

    std::ofstream crisis_out(std::string(QMP_RESULTS_DIR) + "/crisis_labels.csv");
    crisis_out << "name,start,end\n";
    for (const auto& c : crises) crisis_out << c.name << "," << c.start << "," << c.end << "\n";

    // --- Threshold each signal at its own full-sample mean + 2*std (a
    // diagnostic "how unusual is this" cutoff, not a trading rule -- see
    // README for why a full-sample threshold is acceptable here). ---------
    auto flags_and_metrics = [&](const std::string& name, std::function<double(std::size_t)> value_at) {
        std::vector<double> vals;
        for (std::size_t t = dmd_window; t + 1 < T; ++t) vals.push_back(value_at(t));
        double m = mean_of(vals), sd = stdev_of(vals, m);
        double threshold = m + 2.0 * sd;

        std::size_t flagged = 0, flagged_and_crisis = 0, crisis_days = 0, crisis_and_flagged = 0;
        for (std::size_t t = dmd_window; t + 1 < T; ++t) {
            const std::string& date = panel.dates[t + 1];
            bool is_crisis = false;
            for (const auto& c : crises)
                if (in_window(date, c)) is_crisis = true;
            bool is_flagged = value_at(t) > threshold;
            if (is_flagged) ++flagged;
            if (is_crisis) ++crisis_days;
            if (is_flagged && is_crisis) { ++flagged_and_crisis; ++crisis_and_flagged; }
        }
        double precision = flagged > 0 ? static_cast<double>(flagged_and_crisis) / static_cast<double>(flagged) : 0.0;
        double recall = crisis_days > 0 ? static_cast<double>(crisis_and_flagged) / static_cast<double>(crisis_days) : 0.0;

        std::cout << name << ": threshold=" << threshold << " flagged_days=" << flagged << " precision=" << precision
                  << " recall=" << recall << "\n";
        return std::make_tuple(precision, recall, threshold, flagged);
    };

    auto dmd_at = [&](std::size_t t) { return dmd_signal.count(t) ? dmd_signal[t] : 0.0; };
    auto tda_at = [&](std::size_t t) { return tda_signal.count(t) ? tda_signal[t] : 0.0; };
    auto kalman_at = [&](std::size_t t) { return kalman_signal[t]; };

    // Forward-fill DMD/TDA onto the daily grid for fair thresholding.
    {
        double last = 0.0;
        for (std::size_t t = dmd_window; t + 1 < T; ++t) {
            if (dmd_signal.count(t)) last = dmd_signal[t];
            dmd_signal[t] = last;
        }
        last = 0.0;
        for (std::size_t t = dmd_window; t + 1 < T; ++t) {
            if (tda_signal.count(t)) last = tda_signal[t];
            tda_signal[t] = last;
        }
    }

    auto [dmd_p, dmd_r, dmd_thr, dmd_flags] = flags_and_metrics("DMD", dmd_at);
    auto [tda_p, tda_r, tda_thr, tda_flags] = flags_and_metrics("TDA", tda_at);
    auto [kal_p, kal_r, kal_thr, kal_flags] = flags_and_metrics("Kalman", kalman_at);

    std::ofstream pr_out(std::string(QMP_RESULTS_DIR) + "/precision_recall.csv");
    pr_out << "method,precision,recall,threshold,flagged_days\n";
    pr_out << "dmd," << dmd_p << "," << dmd_r << "," << dmd_thr << "," << dmd_flags << "\n";
    pr_out << "tda," << tda_p << "," << tda_r << "," << tda_thr << "," << tda_flags << "\n";
    pr_out << "kalman," << kal_p << "," << kal_r << "," << kal_thr << "," << kal_flags << "\n";

    // --- Lead/lag: first flagged day within [crisis_start - 30d, crisis_end]
    std::ofstream leadlag_out(std::string(QMP_RESULTS_DIR) + "/lead_lag.csv");
    leadlag_out << "crisis,method,first_flag_date,lead_days_before_start\n";
    for (const auto& crisis : crises) {
        for (auto& [method, value_at, thr] :
             std::vector<std::tuple<std::string, std::function<double(std::size_t)>, double>>{
                 {"dmd", dmd_at, dmd_thr}, {"tda", tda_at, tda_thr}, {"kalman", kalman_at, kal_thr}}) {
            std::string first_flag = "none";
            long lead_days = 0;
            for (std::size_t t = dmd_window; t + 1 < T; ++t) {
                const std::string& date = panel.dates[t + 1];
                if (date < crisis.start || date > crisis.end) continue;
                if (value_at(t) > thr) { first_flag = date; break; }
            }
            leadlag_out << crisis.name << "," << method << "," << first_flag << ",n/a\n";
        }
    }

    std::cout << "Wrote results to " << QMP_RESULTS_DIR << "\n";
    return 0;
}
