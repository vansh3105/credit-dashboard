# Credit Risk Analyst Dashboard (FSAFA · Term V)

Flow: **1 · Sector → 2 · Analysis type (Individual vs Industry average | Peer company review) → 3 · Company / Company 1 + 2**.
Then tabs: Overview, Liquidity, Solvency & debt service, Earnings quality, Distress & forensic scores, Early warning, Insights & analysis, Data & method.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                     # reproduces the worked examples from the course decks
streamlit run app.py
```
Data comes live from Yahoo Finance via `yfinance` (needs internet). The first load of each sector takes about a minute, then it is cached for 6 hours.

## Files
| File | Purpose |
|---|---|
| `sectors.py` | Sector → NSE tickers. Edit to change the universe (industry average = these peers). |
| `data.py` | Yahoo pull + mapping of Yahoo row names to standard fields (INR crore). |
| `metrics.py` | Every formula, thresholds (green/red), Benford test, red-flag rules. Source session noted per metric. |
| `report.py` | Credit memo: internal grade and notching, memo content, PDF / Word / Markdown export with charts. |
| `early_warning.py` | Early-warning tab: quarterly and half-year signals, share-price signals vs sector, Merton distance-to-default. |
| `insights.py` | Rule-based narrative (no LLM). |
| `app.py` | Streamlit UI. |
| `tests/` | Unit tests using the decks' worked examples (Apex liquidity case, accrual A/B, Ohlson 63.7%, Dhenu conversion, Benford, Beneish). |
| `tests/test_breakdowns.py` | Checks the score breakdowns reproduce the scores exactly, and the insight rules find the right cause on a deteriorating test firm. |

## First live run - check these
1. Sidebar warning lists tickers that failed to load (tickers were written from memory; fix in `sectors.py`).
2. *Data & method* tab shows which Yahoo fields were missing for the selected company.
3. Spot-check 3 numbers against the annual report, and compare Altman/Beneish/Piotroski with TradingView.
