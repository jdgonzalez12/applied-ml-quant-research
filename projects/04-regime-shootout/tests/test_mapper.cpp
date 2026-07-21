#include <catch2/catch_test_macros.hpp>
#include <queue>
#include <set>
#include <vector>

#include "mapper.hpp"

using namespace qmp04;

TEST_CASE("Mapper never connects two clusters separated by a filter gap", "[mapper]") {
    std::vector<std::vector<double>> points;
    std::vector<double> filter;
    for (double x = -1.0; x <= -0.5 + 1e-9; x += 0.1) { points.push_back({x}); filter.push_back(x); }
    for (double x = 0.5; x <= 1.0 + 1e-9; x += 0.1) { points.push_back({x}); filter.push_back(x); }

    MapperGraph g = compute_mapper(points, filter, 10, 0.1);

    auto side_of = [](const std::vector<double>& pt) { return pt[0] < 0.0 ? -1 : 1; };
    for (const auto& e : g.edges) {
        int side_a = side_of(points[static_cast<std::size_t>(g.nodes[e.a].members[0])]);
        int side_b = side_of(points[static_cast<std::size_t>(g.nodes[e.b].members[0])]);
        REQUIRE(side_a == side_b);
    }
    REQUIRE(g.nodes.size() >= 2);
}

TEST_CASE("Mapper produces a connected graph for a dense, evenly-spaced line", "[mapper]") {
    std::vector<std::vector<double>> points;
    std::vector<double> filter;
    for (int i = 0; i <= 20; ++i) {
        double x = static_cast<double>(i) / 20.0;
        points.push_back({x});
        filter.push_back(x);
    }

    MapperGraph g = compute_mapper(points, filter, 5, 0.5);
    REQUIRE(g.nodes.size() > 1);

    // BFS reachability from node 0 over the edge list.
    std::vector<std::vector<std::size_t>> adj(g.nodes.size());
    for (const auto& e : g.edges) { adj[e.a].push_back(e.b); adj[e.b].push_back(e.a); }

    std::set<std::size_t> visited;
    std::queue<std::size_t> q;
    q.push(0);
    visited.insert(0);
    while (!q.empty()) {
        std::size_t u = q.front();
        q.pop();
        for (std::size_t v : adj[u])
            if (visited.insert(v).second) q.push(v);
    }
    REQUIRE(visited.size() == g.nodes.size());
}
