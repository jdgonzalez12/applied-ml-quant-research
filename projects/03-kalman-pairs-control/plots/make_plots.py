"""Render Project 3's result CSVs into the figures embedded in README.md.
Visualization only: no modeling here."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "common"))
from style import PALETTE, apply_style, savefig  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PLOTS = Path(__file__).resolve().parent
apply_style()


def plot_beta() -> None:
    df = pd.read_csv(RESULTS / "beta_series.csv", parse_dates=["date"])
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["beta_kalman"], color=PALETTE["strategy"], linewidth=1, label="Kalman (adaptive)")
    ax.plot(df["date"], df["beta_static"], color=PALETTE["short"], linewidth=1.5, linestyle="--",
            label="Static OLS (252d training window)")
    ax.set_ylabel(r"Hedge ratio $\beta_t$")
    ax.set_title("CVX ~ XOM hedge ratio: Kalman-adaptive vs. static OLS")
    ax.legend()
    savefig(fig, str(PLOTS / "beta_comparison.png"))
    plt.close(fig)


def plot_signal(tag: str, title: str) -> str:
    df = pd.read_csv(RESULTS / f"signal_{tag}.csv", parse_dates=["date"])
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["zscore"], color=PALETTE["signal"], linewidth=0.7)
    ax.axhline(2.0, color=PALETTE["short"], linestyle="--", linewidth=1, label="entry threshold")
    ax.axhline(-2.0, color=PALETTE["short"], linestyle="--", linewidth=1)
    ax.axhline(0.5, color=PALETTE["long"], linestyle=":", linewidth=1, label="exit threshold")
    ax.axhline(-0.5, color=PALETTE["long"], linestyle=":", linewidth=1)
    ax.set_ylabel("Spread z-score")
    ax.set_title(title)
    ax.legend()
    out = str(PLOTS / f"zscore_{tag}.png")
    savefig(fig, out)
    plt.close(fig)
    return out


def plot_equity_comparison() -> None:
    k = pd.read_csv(RESULTS / "equity_kalman.csv", parse_dates=["date"])
    s = pd.read_csv(RESULTS / "equity_static.csv", parse_dates=["date"])

    fig, ax = plt.subplots()
    ax.plot(k["date"], k["equity"], color=PALETTE["strategy"], label="Kalman-adaptive hedge ratio")
    ax.plot(s["date"], s["equity"], color=PALETTE["short"], linestyle="--", label="Static-OLS hedge ratio")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Pairs-trading equity curve: adaptive vs. static hedge ratio")
    ax.legend()
    savefig(fig, str(PLOTS / "equity_comparison.png"))
    plt.close(fig)


if __name__ == "__main__":
    plot_beta()
    plot_signal("kalman", "Spread z-score (Kalman-adaptive hedge ratio)")
    plot_signal("static", "Spread z-score (static-OLS hedge ratio)")
    plot_equity_comparison()
