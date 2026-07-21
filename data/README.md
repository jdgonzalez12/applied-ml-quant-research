# Data

All data in this directory is fetched once by the scripts in `fetch/` and the
resulting CSVs are committed to the repository, so every project builds and
runs offline, with no API keys, from a fresh clone.

## `raw/equities/` — daily OHLCV bars

**Source:** Yahoo Finance's public `v8/finance/chart` JSON endpoint
(`https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}`), the same
unauthenticated endpoint the `yfinance` library wraps. No API key required.

Stooq's CSV download endpoint (`stooq.com/q/d/l`) was the original choice per
the project plan, but as of this writing it returns an HTML JavaScript
bot-check page instead of CSV for unauthenticated HTTP clients, so the
fetcher was switched to Yahoo Finance. `fetch/fetch_yahoo.py` requests an
explicit `period1`/`period2` range (2005-01-01 through the fetch date)
rather than `range=max`, because Yahoo silently downgrades `range=max`
requests to monthly bars once the span is long enough — an explicit window
keeps true daily resolution.

**Fetched:** 2026-07-21. **Frequency:** daily (trading days only).
**Units:** USD per share for Open/High/Low/Close/AdjClose; shares for
Volume. **Columns:** `Date,Open,High,Low,Close,AdjClose,Volume`.

| File | Instrument | History starts |
|---|---|---|
| `spy.csv` | SPY (S&P 500 ETF) | 2005-01-03 |
| `xlf.csv` | XLF (Financials) | 2005-01-03 |
| `xlk.csv` | XLK (Technology) | 2005-01-03 |
| `xle.csv` | XLE (Energy) | 2005-01-03 |
| `xlv.csv` | XLV (Health Care) | 2005-01-03 |
| `xly.csv` | XLY (Consumer Discretionary) | 2005-01-03 |
| `xlp.csv` | XLP (Consumer Staples) | 2005-01-03 |
| `xli.csv` | XLI (Industrials) | 2005-01-03 |
| `xlu.csv` | XLU (Utilities) | 2005-01-03 |
| `xlb.csv` | XLB (Materials) | 2005-01-03 |
| `xlre.csv` | XLRE (Real Estate) | 2015-10-08 (later inception) |
| `xom.csv` | XOM (Exxon Mobil) | 2005-01-03 |
| `cvx.csv` | CVX (Chevron) | 2005-01-03 |

This universe (SPY + the ten SPDR sector ETFs) is the shared instrument
panel for Project 2 (RMT/PCA statistical arbitrage) and Project 4 (regime
shootout); XOM/CVX is the candidate cointegrated pair for Project 3
(Kalman-filter pairs trading), drawn from the same universe so results are
comparable across projects. The 2005 start date deliberately covers both
the 2008 financial crisis and the 2020 COVID crash for Project 4.

## `raw/macro/` — FRED series

**Source:** FRED's public `fredgraph.csv` endpoint
(`https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}`). No API
key required. **Fetched:** 2026-07-21. **Frequency:** daily (as published;
Treasury series are business-day, VIX is daily).

| File | Series ID | Description | Units |
|---|---|---|---|
| `dgs10.csv` | DGS10 | 10-Year Treasury Constant Maturity Rate | Percent |
| `dgs2.csv` | DGS2 | 2-Year Treasury Constant Maturity Rate | Percent |
| `vixcls.csv` | VIXCLS | CBOE Volatility Index (VIX), close | Index points |
| `t10y2y.csv` | T10Y2Y | 10-Year minus 2-Year Treasury yield spread | Percentage points |

Used as macro/stress context in Project 4's regime shootout (VIX and the
10y-2y spread are both standard crisis proxies) and as a documented data
type outside pure equity returns.

## Regenerating

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python data\fetch\fetch_yahoo.py
.venv\Scripts\python data\fetch\fetch_fred.py
```

Both scripts are idempotent (re-running overwrites the same files) and
perform no computation — they only fetch and write CSV.
