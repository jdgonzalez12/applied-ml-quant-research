"""Download daily OHLCV bars from Yahoo Finance's public chart endpoint (no
API key required) and cache them as CSV under data/raw/equities/.

Stooq's CSV download endpoint now sits behind a JavaScript bot-check and is
no longer reachable with a plain HTTP client, so this project uses Yahoo's
`v8/finance/chart` JSON endpoint instead -- the same public, unauthenticated
endpoint the `yfinance` library wraps. Pure I/O: no modeling or statistics.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "equities"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# SPY + the SPDR sector ETFs: the shared universe used by Project 2 (RMT/PCA)
# and Project 4 (regime shootout), plus the XOM/CVX pair used by Project 3.
TICKERS = [
    "SPY", "XLF", "XLK", "XLE", "XLV", "XLY",
    "XLP", "XLI", "XLU", "XLB", "XLRE",
    "XOM", "CVX",
]


PERIOD1 = int(dt.datetime(2005, 1, 1, tzinfo=dt.timezone.utc).timestamp())


def fetch_one(symbol: str) -> None:
    # Explicit period1/period2 (not range=max): requesting "max" range with
    # interval=1d silently gets downgraded by Yahoo to monthly bars once the
    # span is long enough. An explicit start date keeps true daily
    # resolution, which Project 4's regime shootout needs across both the
    # 2008 GFC and the 2020 COVID crash.
    period2 = int(dt.datetime.now(dt.timezone.utc).timestamp())
    params = {"period1": PERIOD1, "period2": period2, "interval": "1d", "events": "div,split"}
    resp = requests.get(CHART_URL.format(symbol=symbol), params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    result = payload.get("chart", {}).get("result")
    if not result:
        print(f"  WARNING: {symbol} returned no chart result", file=sys.stderr)
        return
    r0 = result[0]
    timestamps = r0.get("timestamp", [])
    quote = r0["indicators"]["quote"][0]
    adjclose = r0["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(timestamps))

    lines = ["Date,Open,High,Low,Close,AdjClose,Volume"]
    for i, ts in enumerate(timestamps):
        date = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        row = [date] + [
            "" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
            for v in (quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i], adjclose[i], quote["volume"][i])
        ]
        if row[1] == "":  # skip non-trading gaps with null OHLC
            continue
        lines.append(",".join(row))

    out_path = OUT_DIR / f"{symbol.lower()}.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path.relative_to(OUT_DIR.parents[1])} ({len(lines) - 1} rows)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        fetch_one(ticker)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
