#pragma once

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libquant/matrix.hpp"

namespace qmp04 {

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

    PricePanel panel;
    panel.symbols = symbols;
    panel.dates = common;
    panel.prices = libquant::Matrix<double>(common.size(), symbols.size());
    for (std::size_t t = 0; t < common.size(); ++t)
        for (std::size_t j = 0; j < symbols.size(); ++j)
            panel.prices(t, j) = series[j].at(common[t]);
    return panel;
}

inline libquant::Matrix<double> log_returns(const libquant::Matrix<double>& prices) {
    const std::size_t T = prices.rows(), N = prices.cols();
    libquant::Matrix<double> r(T - 1, N);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t t = 1; t < T; ++t)
            r(t - 1, j) = std::log(prices(t, j) / prices(t - 1, j));
    return r;
}

// Reads a FRED CSV (DATE,SERIES_ID header) into {date -> value}, skipping
// missing/"." entries.
inline std::map<std::string, double> load_fred_series(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("csv_loader: cannot open " + path);
    std::map<std::string, double> out;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, value;
        std::getline(ss, date, ',');
        std::getline(ss, value, ',');
        if (date.empty() || value.empty() || value == ".") continue;
        out[date] = std::stod(value);
    }
    return out;
}

}  // namespace qmp04
