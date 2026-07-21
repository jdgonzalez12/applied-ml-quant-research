#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace qmp04 {

struct PersistencePair {
    int dimension = 0;
    double birth = 0.0;
    double death = 0.0;
};

struct SimplexInfo {
    std::vector<int> vertices;  // sorted
    double value = 0.0;
    int dim = 0;
    std::vector<int> boundary;  // indices into the final simplex list, of one-lower-dimensional faces
};

namespace detail {
struct RawSimplex {
    std::vector<int> vertices;
    double value;
    int dim;
};
}  // namespace detail

// Builds the Vietoris-Rips complex (dimensions 0, 1, 2) up to max_epsilon
// from a point cloud (Carlsson & Vejdemo-Johansson, "Topological Data
// Analysis with Applications"). Simplices are returned in a valid
// filtration order (every face precedes its cofaces): sorted by filtration
// value ascending, dimension ascending as a tiebreak -- which is exactly
// what a face/coface relationship guarantees (a triangle's value is the max
// of its 3 edges, so it can never sort before any of them).
inline std::vector<SimplexInfo> build_vietoris_rips(const std::vector<std::vector<double>>& points,
                                                     double max_epsilon) {
    const std::size_t n = points.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < points[i].size(); ++k) {
                double d = points[i][k] - points[j][k];
                s += d * d;
            }
            D[i][j] = D[j][i] = std::sqrt(s);
        }

    std::vector<detail::RawSimplex> raw;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (D[i][j] <= max_epsilon) raw.push_back({{static_cast<int>(i), static_cast<int>(j)}, D[i][j], 1});

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (D[i][j] > max_epsilon) continue;
            for (std::size_t k = j + 1; k < n; ++k) {
                if (D[i][k] > max_epsilon || D[j][k] > max_epsilon) continue;
                double val = std::max({D[i][j], D[i][k], D[j][k]});
                raw.push_back({{static_cast<int>(i), static_cast<int>(j), static_cast<int>(k)}, val, 2});
            }
        }
    }

    std::sort(raw.begin(), raw.end(), [](const detail::RawSimplex& a, const detail::RawSimplex& b) {
        if (a.value != b.value) return a.value < b.value;
        return a.dim < b.dim;
    });

    std::vector<SimplexInfo> simplices;
    simplices.reserve(n + raw.size());
    std::map<std::vector<int>, int> index_of;
    for (std::size_t i = 0; i < n; ++i) {
        SimplexInfo s;
        s.vertices = {static_cast<int>(i)};
        s.value = 0.0;
        s.dim = 0;
        index_of[s.vertices] = static_cast<int>(simplices.size());
        simplices.push_back(s);
    }
    for (auto& r : raw) {
        SimplexInfo s;
        s.vertices = r.vertices;
        s.value = r.value;
        s.dim = r.dim;
        if (r.dim == 1) {
            s.boundary = {index_of[{r.vertices[0]}], index_of[{r.vertices[1]}]};
        } else {
            std::vector<int> e01 = {r.vertices[0], r.vertices[1]};
            std::vector<int> e02 = {r.vertices[0], r.vertices[2]};
            std::vector<int> e12 = {r.vertices[1], r.vertices[2]};
            s.boundary = {index_of[e01], index_of[e02], index_of[e12]};
        }
        index_of[s.vertices] = static_cast<int>(simplices.size());
        simplices.push_back(s);
    }
    return simplices;
}

// Standard persistence algorithm (Edelsbrunner-Letscher-Zomorodian /
// Zomorodian-Carlsson) over Z/2: reduces the boundary matrix by left-to-
// right column operations, tracking the lowest (highest-index) nonzero row
// per column. When column j reduces to a nonzero column with low index i,
// simplex i (a birth) is paired with simplex j (its death): a homology
// class of dimension dim(i) is born at filtration value(i) and dies at
// value(j). Unpaired columns represent classes that never die (infinite
// persistence) and are not returned here, since this project only uses
// finite-persistence H1 features as the topological regime signal.
inline std::vector<PersistencePair> compute_persistence(const std::vector<SimplexInfo>& simplices) {
    const std::size_t m = simplices.size();
    std::vector<std::vector<int>> columns(m);
    for (std::size_t j = 0; j < m; ++j) {
        columns[j] = simplices[j].boundary;
        std::sort(columns[j].begin(), columns[j].end());
    }

    auto low_of = [](const std::vector<int>& col) { return col.empty() ? -1 : col.back(); };
    auto sym_diff = [](const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> out;
        std::size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] == b[j]) { ++i; ++j; }
            else if (a[i] < b[j]) out.push_back(a[i++]);
            else out.push_back(b[j++]);
        }
        while (i < a.size()) out.push_back(a[i++]);
        while (j < b.size()) out.push_back(b[j++]);
        return out;
    };

    std::vector<int> pivot_to_col(m, -1);
    std::vector<PersistencePair> pairs;

    for (std::size_t j = 0; j < m; ++j) {
        while (true) {
            int lo = low_of(columns[j]);
            if (lo < 0) break;
            if (pivot_to_col[static_cast<std::size_t>(lo)] == -1) {
                pivot_to_col[static_cast<std::size_t>(lo)] = static_cast<int>(j);
                break;
            }
            columns[j] = sym_diff(columns[j], columns[static_cast<std::size_t>(pivot_to_col[static_cast<std::size_t>(lo)])]);
        }
        int lo = low_of(columns[j]);
        if (lo >= 0) {
            const auto& birth_simplex = simplices[static_cast<std::size_t>(lo)];
            pairs.push_back({birth_simplex.dim, birth_simplex.value, simplices[j].value});
        }
    }
    return pairs;
}

// Largest H1 (loop) persistence: death - birth over all dimension-1 pairs,
// 0 if there are none. This is the scalar topological-complexity signal
// tracked over rolling windows as a regime-detection statistic.
inline double max_h1_persistence(const std::vector<PersistencePair>& pairs) {
    double best = 0.0;
    for (const auto& p : pairs)
        if (p.dimension == 1) best = std::max(best, p.death - p.birth);
    return best;
}

}  // namespace qmp04
