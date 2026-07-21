"""Render Project 2's result CSVs (produced by the C++ pipeline) into the
figures embedded in README.md. Visualization only: no modeling here."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "common"))
from style import PALETTE, apply_style, savefig  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PLOTS = Path(__file__).resolve().parent
apply_style()


def plot_eigen_spectrum() -> None:
    df = pd.read_csv(RESULTS / "eigen_spectrum.csv")
    first_date = df["window_end_date"].iloc[0]
    window = df[df["window_end_date"] == first_date].sort_values("eigenvalue_rank")
    lam_plus = window["lambda_plus"].iloc[0]
    lam_minus = window["lambda_minus"].iloc[0]

    # Theoretical Marchenko-Pastur density for q = N/T (sigma^2 = 1, since
    # the correlation matrix has unit-variance standardized inputs).
    q = 9.0 / 60.0
    lam = np.linspace(max(lam_minus, 1e-4), lam_plus, 400)
    density = np.sqrt(np.clip((lam_plus - lam) * (lam - lam_minus), 0, None)) / (2 * np.pi * q * lam)
    density_scaled = density / density.max() * 0.9  # scaled for visual overlay against the eigenvalue stem plot

    fig, ax = plt.subplots()
    ax.vlines(window["eigenvalue"], 0, 1, color=PALETTE["signal"], linewidth=2, label="Empirical eigenvalues")
    ax.plot(lam, density_scaled, color=PALETTE["accent"], linewidth=2, label="Marchenko-Pastur density (scaled)")
    ax.axvline(lam_plus, color=PALETTE["short"], linestyle="--", linewidth=1, label=r"$\lambda_+$ (MP edge)")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("(normalized)")
    ax.set_title(f"Correlation matrix eigenvalue spectrum vs. Marchenko-Pastur bulk ({first_date})")
    ax.legend()
    savefig(fig, str(PLOTS / "eigen_spectrum.png"))
    plt.close(fig)


def plot_factor_loadings() -> None:
    df = pd.read_csv(RESULTS / "factor_loadings.csv")
    last_date = df["window_end_date"].iloc[-1]
    snap = df[(df["window_end_date"] == last_date) & (df["pc"] == 1)].sort_values("loading")

    fig, ax = plt.subplots()
    colors = [PALETTE["long"] if v >= 0 else PALETTE["short"] for v in snap["loading"]]
    ax.barh(snap["symbol"].str.upper(), snap["loading"], color=colors)
    ax.set_xlabel("PC1 (market mode) loading")
    ax.set_title(f"First eigenportfolio loadings by sector ETF ({last_date})")
    savefig(fig, str(PLOTS / "factor_loadings.png"))
    plt.close(fig)


def plot_sscore() -> None:
    df = pd.read_csv(RESULTS / "sscore.csv", parse_dates=["date"])
    symbol = "xlk"  # representative example series
    s = df[df["symbol"] == symbol].sort_values("date")

    fig, ax = plt.subplots()
    ax.plot(s["date"], s["s_score"], color=PALETTE["signal"], linewidth=0.8)
    ax.axhline(1.25, color=PALETTE["short"], linestyle="--", linewidth=1, label="entry threshold")
    ax.axhline(-1.25, color=PALETTE["short"], linestyle="--", linewidth=1)
    ax.axhline(0.5, color=PALETTE["long"], linestyle=":", linewidth=1, label="exit threshold")
    ax.axhline(-0.5, color=PALETTE["long"], linestyle=":", linewidth=1)
    ax.set_ylabel("s-score (equilibrium std. deviations)")
    ax.set_title(f"OU s-score for {symbol.upper()} residual, with entry/exit thresholds")
    ax.legend()
    savefig(fig, str(PLOTS / "sscore.png"))
    plt.close(fig)


def plot_equity_curve() -> None:
    df = pd.read_csv(RESULTS / "equity_curve.csv", parse_dates=["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df["date"], df["strategy_equity"], color=PALETTE["strategy"], label="RMT/PCA stat-arb (net of costs)")
    ax1.plot(df["date"], df["spy_equity"], color=PALETTE["benchmark"], label="SPY buy & hold", linewidth=1)
    ax1.set_ylabel("Growth of $1")
    ax1.set_title("Strategy vs. SPY buy-and-hold")
    ax1.legend()

    running_peak = df["strategy_equity"].cummax()
    drawdown = (df["strategy_equity"] - running_peak) / running_peak
    ax2.fill_between(df["date"], drawdown, 0, color=PALETTE["short"], alpha=0.5)
    ax2.set_ylabel("Drawdown")

    savefig(fig, str(PLOTS / "equity_curve.png"))
    plt.close(fig)


if __name__ == "__main__":
    plot_eigen_spectrum()
    plot_factor_loadings()
    plot_sscore()
    plot_equity_curve()
