#pragma once

#include <cstddef>
#include <vector>

namespace libquant_backtest {

// Applies fn(start, end) for each fixed-size sliding window [start, end) of
// length `window`, advancing by `step`, and collects the per-window results.
// A single shared, tested iteration pattern for rolling re-estimation
// (rolling PCA/RMT, rolling Kalman re-init, rolling DMD/TDA windows) used
// identically across projects 2-4.
template <typename Result, typename Fn>
std::vector<Result> rolling_window_run(std::size_t n, std::size_t window, std::size_t step, Fn&& fn) {
    std::vector<Result> results;
    if (window == 0 || step == 0 || window > n) return results;
    for (std::size_t start = 0; start + window <= n; start += step) results.push_back(fn(start, start + window));
    return results;
}

}  // namespace libquant_backtest
