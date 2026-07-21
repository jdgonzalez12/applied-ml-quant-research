"""Render Project 4's result CSVs into the figures embedded in README.md.
Visualization only: no modeling here -- lead/lag is a plain date
subtraction of C++-produced dates, not a statistical computation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "common"))
from style import PALETTE, apply_style, savefig  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PLOTS = Path(__file__).resolve().parent
apply_style()


def plot_timeline() -> None:
    df = pd.read_csv(RESULTS / "regime_signals.csv", parse_dates=["date"])
    crises = pd.read_csv(RESULTS / "crisis_labels.csv", parse_dates=["start", "end"])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    series = [
        ("dmd_growth_rate", "DMD max growth rate", PALETTE["strategy"]),
        ("tda_persistence", "TDA normalized H1 persistence", PALETTE["accent"]),
        ("kalman_innovation", "Kalman |standardized innovation|", PALETTE["signal"]),
    ]
    for ax, (col, title, color) in zip(axes, series):
        ax.plot(df["date"], df[col], color=color, linewidth=0.8)
        for _, c in crises.iterrows():
            ax.axvspan(c["start"], c["end"], color=PALETTE["short"], alpha=0.15)
        ax.set_title(title, fontsize=10, loc="left")
    fig.suptitle("Regime-detection signals vs. hand-labeled crisis windows (shaded)", y=1.0)
    savefig(fig, str(PLOTS / "signal_timeline.png"))
    plt.close(fig)


def plot_precision_recall() -> None:
    df = pd.read_csv(RESULTS / "precision_recall.csv")
    fig, ax = plt.subplots()
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["precision"], width, label="Precision", color=PALETTE["strategy"])
    ax.bar([i + width / 2 for i in x], df["recall"], width, label="Recall", color=PALETTE["accent"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"].str.upper())
    ax.set_title("Precision / recall vs. hand-labeled crisis days (threshold = mean + 2 std)")
    ax.legend()
    savefig(fig, str(PLOTS / "precision_recall.png"))
    plt.close(fig)


def compute_lead_lag_table() -> pd.DataFrame:
    ll = pd.read_csv(RESULTS / "lead_lag.csv")
    crises = pd.read_csv(RESULTS / "crisis_labels.csv")
    crisis_start = dict(zip(crises["name"], pd.to_datetime(crises["start"])))

    rows = []
    for _, r in ll.iterrows():
        if r["first_flag_date"] == "none":
            rows.append({"crisis": r["crisis"], "method": r["method"], "lag_days": None})
        else:
            flag = pd.to_datetime(r["first_flag_date"])
            start = crisis_start[r["crisis"]]
            rows.append({"crisis": r["crisis"], "method": r["method"], "lag_days": (flag - start).days})
    return pd.DataFrame(rows)


def plot_mapper_graph() -> None:
    nodes = pd.read_csv(RESULTS / "mapper_nodes.csv")
    edges = pd.read_csv(RESULTS / "mapper_edges.csv")

    G = nx.Graph()
    for _, n in nodes.iterrows():
        G.add_node(int(n["node_id"]), size=n["size"], crisis_fraction=n["crisis_fraction"],
                    filter_center=n["filter_center"])
    for _, e in edges.iterrows():
        G.add_edge(int(e["node_a"]), int(e["node_b"]), weight=e["shared"])

    # Layout: x-position follows the filter value directly (the natural
    # "spine" of a Mapper graph built from a scalar filter), y from a
    # spring layout for readability -- a hybrid layout used in most
    # published Mapper visualizations.
    pos = nx.spring_layout(G, seed=7, k=0.6)
    for nid in G.nodes:
        x, _ = pos[nid]
        pos[nid] = (G.nodes[nid]["filter_center"], x)

    sizes = [30 + 4 * G.nodes[n]["size"] ** 0.5 for n in G.nodes]
    colors = [G.nodes[n]["crisis_fraction"] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=PALETTE["benchmark"], alpha=0.5, width=1)
    sc = ax.scatter([pos[n][0] for n in G.nodes], [pos[n][1] for n in G.nodes], s=sizes, c=colors,
                     cmap="RdYlGn_r", vmin=0, vmax=1, edgecolors="black", linewidths=0.5, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Fraction of days in a hand-labeled crisis window")
    ax.set_xlabel(r"Filter value: $\|\mathrm{daily\ sector\ return\ vector}\|_2$ (market-stress proxy)")
    ax.set_yticks([])
    ax.set_title("Mapper graph of daily sector-return dynamics, 2005-2026")
    savefig(fig, str(PLOTS / "mapper_graph.png"))
    plt.close(fig)


if __name__ == "__main__":
    plot_timeline()
    plot_precision_recall()
    table = compute_lead_lag_table()
    table.to_csv(RESULTS / "lead_lag_days.csv", index=False)
    print(table.to_string(index=False))
    plot_mapper_graph()
