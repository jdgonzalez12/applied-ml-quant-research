"""Shared matplotlib theme for every project's make_plots.py.

Visualization only: these helpers read CSV artifacts already computed by
the C++ pipelines and format figures. No modeling or statistics live here.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "signal": "#1f77b4",
    "noise": "#7f7f7f",
    "long": "#2ca02c",
    "short": "#d62728",
    "benchmark": "#7f7f7f",
    "strategy": "#1f77b4",
    "accent": "#ff7f0e",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "figure.dpi": 130,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def savefig(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")
