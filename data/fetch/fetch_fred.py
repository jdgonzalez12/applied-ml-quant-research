"""Download macro/rates series from FRED's public CSV endpoint (no API key
required) and cache them under data/raw/macro/. Pure I/O: no modeling here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "macro"

# 10y/2y Treasury yields (yield-curve context for Project 4), VIX (stress
# proxy), and the 10y-2y spread series used directly as a crisis indicator.
SERIES = ["DGS10", "DGS2", "VIXCLS", "T10Y2Y"]


def fetch_one(series_id: str) -> None:
    url = FRED_URL.format(series_id=series_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or "DATE" not in text.splitlines()[0].upper():
        print(f"  WARNING: {series_id} returned unexpected content", file=sys.stderr)
        return
    out_path = OUT_DIR / f"{series_id.lower()}.csv"
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"  wrote {out_path.relative_to(OUT_DIR.parents[1])} ({len(text.splitlines())} rows)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for series_id in SERIES:
        fetch_one(series_id)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
