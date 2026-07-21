"""Aggregate Projects 2-4's result CSVs (already computed by their C++
pipelines) into one cross-project comparison table and plot. Visualization
and simple table joins only -- no modeling."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "common"))
from style import PALETTE, apply_style, savefig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)
apply_style()


def build_strategy_table() -> pd.DataFrame:
    p2 = pd.read_csv(ROOT / "projects/02-rmt-pca-stat-arb/results/metrics.csv").set_index("metric")["value"]
    p3 = pd.read_csv(ROOT / "projects/03-kalman-pairs-control/results/metrics.csv")

    rows = [
        {
            "strategy": "P2: RMT/PCA stat-arb",
            "sharpe": p2["sharpe"],
            "max_drawdown": p2["max_drawdown"],
            "hit_rate": p2["hit_rate"],
            "annualized_return": p2["annualized_return"],
        }
    ]
    for _, r in p3.iterrows():
        label = "P3: Kalman-adaptive pairs" if r["method"] == "kalman" else "P3: Static-OLS pairs (baseline)"
        rows.append({
            "strategy": label,
            "sharpe": r["sharpe"],
            "max_drawdown": r["max_drawdown"],
            "hit_rate": r["hit_rate"],
            "annualized_return": r["annualized_return"],
        })
    return pd.DataFrame(rows)


def plot_strategy_comparison(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = [PALETTE["strategy"], PALETTE["long"], PALETTE["short"]]

    ax1.bar(df["strategy"], df["sharpe"], color=colors)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_title("Sharpe ratio by strategy")
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(df["strategy"], df["max_drawdown"] * 100, color=colors)
    ax2.set_ylabel("Max drawdown (%)")
    ax2.set_title("Max drawdown by strategy")
    ax2.tick_params(axis="x", rotation=20)

    savefig(fig, str(PLOTS / "strategy_comparison.png"))
    plt.close(fig)


def build_detection_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / "projects/04-regime-shootout/results/precision_recall.csv")


if __name__ == "__main__":
    strategy_df = build_strategy_table()
    strategy_df.to_csv(Path(__file__).resolve().parent / "strategy_comparison.csv", index=False)
    print(strategy_df.to_string(index=False))
    plot_strategy_comparison(strategy_df)

    detection_df = build_detection_table()
    print()
    print(detection_df.to_string(index=False))
