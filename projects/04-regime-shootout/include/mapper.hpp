#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

namespace qmp04 {

struct MapperNode {
    std::size_t interval = 0;
    std::vector<int> members;  // indices into the original point set
    double filter_center = 0.0;
};

struct MapperEdge {
    std::size_t a = 0, b = 0;  // node indices
    std::size_t shared = 0;
};

struct MapperGraph {
    std::vector<MapperNode> nodes;
    std::vector<MapperEdge> edges;
};

// The Mapper algorithm (Singh, Memoli, Carlsson 2007; Carlsson &
// Vejdemo-Johansson, "Topological Data Analysis with Applications"): a
// combinatorial summary of a point cloud's shape.
//   1. Cover the range of a scalar filter function f with `resolution`
//      overlapping intervals (overlap fraction `overlap`).
//   2. For each interval, cluster the points whose f-value falls inside it
//      -- single-linkage via union-find, cut at a per-interval adaptive
//      distance threshold (the interval's own median pairwise distance,
//      the same k-NN-style sparsification heuristic used for the
//      persistent-homology module).
//   3. Each (interval, cluster) pair with >=1 point becomes a graph node;
//      two nodes get an edge if their point sets overlap (which can only
//      happen for nodes from overlapping intervals, since the cover's
//      overlap is exactly what lets a point belong to two intervals at
//      once).
// Unlike persistent homology's precise birth/death pairs, Mapper trades
// exactness for an intuitive, directly visualizable "shape graph" -- used
// here as a qualitative regime-detection exploration, not a scored method.
inline MapperGraph compute_mapper(const std::vector<std::vector<double>>& points, const std::vector<double>& filter,
                                   std::size_t resolution, double overlap) {
    MapperGraph graph;
    const std::size_t n = points.size();
    if (n == 0) return graph;

    double fmin = *std::min_element(filter.begin(), filter.end());
    double fmax = *std::max_element(filter.begin(), filter.end());
    if (fmax <= fmin) fmax = fmin + 1.0;

    double span = (fmax - fmin) / static_cast<double>(resolution);
    double half_width = span * (1.0 + overlap) / 2.0;

    for (std::size_t iv = 0; iv < resolution; ++iv) {
        double center = fmin + span * (static_cast<double>(iv) + 0.5);
        double lo = center - half_width, hi = center + half_width;

        std::vector<int> members;
        for (std::size_t i = 0; i < n; ++i)
            if (filter[i] >= lo && filter[i] <= hi) members.push_back(static_cast<int>(i));
        if (members.empty()) continue;

        // Single-linkage clustering within this interval's preimage, cut at
        // the interval's own median pairwise distance.
        const std::size_t m = members.size();
        std::vector<double> dists;
        auto dist = [&](int a, int b) {
            double s = 0.0;
            for (std::size_t d = 0; d < points[static_cast<std::size_t>(a)].size(); ++d) {
                double diff = points[static_cast<std::size_t>(a)][d] - points[static_cast<std::size_t>(b)][d];
                s += diff * diff;
            }
            return std::sqrt(s);
        };
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = i + 1; j < m; ++j) dists.push_back(dist(members[i], members[j]));

        std::vector<std::size_t> parent(m);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<std::size_t(std::size_t)> find = [&](std::size_t x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };

        if (!dists.empty()) {
            std::vector<double> sorted_d = dists;
            std::sort(sorted_d.begin(), sorted_d.end());
            double threshold = sorted_d[sorted_d.size() / 2];
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = i + 1; j < m; ++j)
                    if (dist(members[i], members[j]) <= threshold) {
                        std::size_t ri = find(i), rj = find(j);
                        if (ri != rj) parent[ri] = rj;
                    }
        }

        std::map<std::size_t, std::vector<int>> clusters;
        for (std::size_t i = 0; i < m; ++i) clusters[find(i)].push_back(members[i]);

        for (auto& [root, pts] : clusters) {
            MapperNode node;
            node.interval = iv;
            node.members = pts;
            node.filter_center = center;
            graph.nodes.push_back(node);
        }
    }

    for (std::size_t a = 0; a < graph.nodes.size(); ++a) {
        for (std::size_t b = a + 1; b < graph.nodes.size(); ++b) {
            if (graph.nodes[a].interval == graph.nodes[b].interval) continue;
            std::size_t shared = 0;
            for (int pa : graph.nodes[a].members)
                for (int pb : graph.nodes[b].members)
                    if (pa == pb) ++shared;
            if (shared > 0) graph.edges.push_back({a, b, shared});
        }
    }

    return graph;
}

}  // namespace qmp04
