#pragma once

#include <cstddef>
#include <vector>

namespace libquant_backtest {

struct WalkForwardSplit {
    std::size_t train_start, train_end;  // half-open [train_start, train_end)
    std::size_t test_start, test_end;    // half-open [test_start, test_end)
};

// Expanding-window walk-forward splitter: train on [0, train_end), test on
// the following `test_size` points, advance by `step`, repeat. Every test
// window strictly follows the data used to fit it, which is what rules out
// look-ahead bias by construction rather than by convention.
inline std::vector<WalkForwardSplit> walk_forward_splits(std::size_t n, std::size_t initial_train,
                                                          std::size_t test_size, std::size_t step = 0) {
    if (step == 0) step = test_size;
    std::vector<WalkForwardSplit> splits;
    if (test_size == 0 || step == 0) return splits;
    std::size_t train_end = initial_train;
    while (train_end + test_size <= n) {
        splits.push_back(WalkForwardSplit{0, train_end, train_end, train_end + test_size});
        train_end += step;
    }
    return splits;
}

}  // namespace libquant_backtest
