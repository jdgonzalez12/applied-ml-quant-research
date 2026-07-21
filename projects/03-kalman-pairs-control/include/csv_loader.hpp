#pragma once

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libquant/matrix.hpp"

namespace qmp03 {

// Reads the Date,Open,High,Low,Close,AdjClose,Volume CSVs produced by
// data/fetch/fetch_yahoo.py and returns {date -> adjusted close}.
inline std::map<std::string, double> load_adjclose(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("csv_loader: cannot open " + path);
    std::map<std::string, double> out;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, open, high, low, close, adjclose, volume;
        std::getline(ss, date, ',');
        std::getline(ss, open, ',');
        std::getline(ss, high, ',');
        std::getline(ss, low, ',');
        std::getline(ss, close, ',');
        std::getline(ss, adjclose, ',');
        std::getline(ss, volume, ',');
        if (date.empty() || adjclose.empty()) continue;
        out[date] = std::stod(adjclose);
    }
    return out;
}

// Inner-joins adjusted-close series for `symbols` (files "<symbol>.csv" in
// `dir`) on their common trading dates and returns a T x N price matrix
// (rows = dates ascending, columns = symbols, in the given order) plus the
// aligned date labels.
struct PricePanel {
    libquant::Matrix<double> prices;
    std::vector<std::string> dates;
    std::vector<std::string> symbols;
};

inline PricePanel load_price_panel(const std::string& dir, const std::vector<std::string>& symbols) {
    std::vector<std::map<std::string, double>> series;
    series.reserve(symbols.size());
    for (const auto& sym : symbols) series.push_back(load_adjclose(dir + "/" + sym + ".csv"));

    std::vector<std::string> common;
    for (const auto& [date, _] : series.front()) {
        bool in_all = true;
        for (std::size_t i = 1; i < series.size(); ++i) {
            if (series[i].find(date) == series[i].end()) { in_all = false; break; }
        }
        if (in_all) common.push_back(date);
    }
    // std::map keys are already sorted ascending (ISO date strings sort correctly).

    PricePanel panel;
    panel.symbols = symbols;
    panel.dates = common;
    panel.prices = libquant::Matrix<double>(common.size(), symbols.size());
    for (std::size_t t = 0; t < common.size(); ++t)
        for (std::size_t j = 0; j < symbols.size(); ++j)
            panel.prices(t, j) = series[j].at(common[t]);
    return panel;
}

}  // namespace qmp03
