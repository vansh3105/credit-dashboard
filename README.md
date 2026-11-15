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
| `news.py` | News & ratings tab: Google News / Yahoo headlines, keyword credit screen, rating extraction and mapping to CR grades, optional Claude analysis. |
| `snapshot.py` | Reads the nightly data snapshot (JSON, from the repo's `data` branch) and compares the last two nights. |
| `scripts/build_snapshot.py` | Builds the snapshot and daily status history; run nightly by GitHub Actions. |
| `early_warning.py` | Early-warning tab: quarterly and half-year signals, share-price signals vs sector, Merton distance-to-default. |
| `insights.py` | Rule-based narrative (no LLM). |
| `app.py` | Streamlit UI. |
| `tests/` | Unit tests using the decks' worked examples (Apex liquidity case, accrual A/B, Ohlson 63.7%, Dhenu conversion, Benford, Beneish). |
| `tests/test_news.py` | Keyword screen, rating extraction and attribution, news signal, memo rendering, and the AI call with a fake client. |
| `tests/test_snapshot.py` | Snapshot round-trip is lossless, freshness and fallback order, and the what-changed comparison. |
| `tests/test_breakdowns.py` | Checks the score breakdowns reproduce the scores exactly, and the insight rules find the right cause on a deteriorating test firm. |

## First live run - check these
1. Sidebar warning lists tickers that failed to load (tickers were written from memory; fix in `sectors.py`).
2. *Data & method* tab shows which Yahoo fields were missing for the selected company.
3. Spot-check 3 numbers against the annual report, and compare Altman/Beneish/Piotroski with TradingView.

## Optional: AI reading of the news

The News & ratings tab can ask Claude to read the headlines. Add an Anthropic API key:

- **Streamlit Cloud:** app menu → Settings → Secrets → `ANTHROPIC_API_KEY = "sk-ant-..."`
- **Locally:** create `.streamlit/secrets.toml` with the same line (git-ignored).

Without a key everything else works; the AI never changes the grade or the early-warning signal.

## How the live app stays fast and reliable

- **Nightly snapshot.** `.github/workflows/snapshot.yml` runs at 06:00 IST, fetches all 203 companies (statements,
  quarters, prices, news) and publishes `snapshot.json.gz` and `history.csv` to the `data` branch. The app reads it
  first (about 2 s to open instead of about 12 s). *Reload data* in the sidebar skips it and fetches live.
- **Fallback.** If Yahoo or Google News fail, the app shows the last good nightly copy and says so in the sidebar.
- **What changed.** The Overview tab compares the last two nights: grade, headline signal, early-warning level,
  news flow and score moves of 5+ points.
- **Tested before it goes live.** `.github/workflows/tests.yml` runs the tests on Python 3.11 and 3.13 on every push
  to `main`, then moves the `live` branch to that commit only if they pass. Point Streamlit Cloud at `live`.
- **Pinned versions.** `requirements.txt` pins exact library versions so an upstream release cannot break the app.
