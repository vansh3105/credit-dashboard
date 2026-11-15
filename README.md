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
Data comes from Yahoo Finance via `yfinance`, served from a nightly snapshot (opens in about 2 s); *Reload data* fetches live. Live results are cached for 6 hours.

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
| `macro.py` | Live risk-free rate (RBI 364-day T-bill, OECD fallback) and SEBI enforcement orders (RSS, archived nightly). |
| `banks.py` | Banks & NBFCs: lender ratios (capital, asset quality, earnings, efficiency), derived credit cost, manual NPA / CRAR. |
| `scenario.py` | What-if on the latest year (revenue, margin, rate shocks) and the revenue fall at which cover breaks. |
| `groups.py` | Business groups (Tata, Adani, JSW, PSUs…) for group-exposure checks. |
| `scripts/alerts.py` | Nightly watchlist summary and optional email. |
| `early_warning.py` | Early-warning tab: quarterly and half-year signals, share-price signals vs sector, Merton distance-to-default. |
| `insights.py` | Rule-based narrative (no LLM). |
| `app.py` | Streamlit UI. |
| `tests/` | Unit tests using the decks' worked examples (Apex liquidity case, accrual A/B, Ohlson 63.7%, Dhenu conversion, Benford, Beneish). |
| `tests/test_news.py` | Keyword screen, rating extraction and attribution, news signal, memo rendering, and the AI call with a fake client. |
| `tests/test_snapshot.py` | Snapshot round-trip is lossless, freshness and fallback order, and the what-changed comparison. |
| `tests/test_macro.py` | RBI / OECD rate parsing and fallback order, SEBI archive merge, matching and severity (offline). |
| `tests/test_features.py` | Score attribution, what-if and break points, bank ratios, groups, pledge notch, data gap fills, alert report. |
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
- **Tested before it goes live.** Changes are pushed to the `dev` branch. `.github/workflows/tests.yml` runs the tests
  on Python 3.11 and 3.13 and moves `main` - the branch Streamlit Cloud deploys - to that commit only if they pass.
  If a run fails, the live app simply stays on the last good version (see the Actions tab on GitHub).
- **Pinned versions.** `requirements.txt` pins exact library versions so an upstream release cannot break the app.

## Daily alert email (optional)

1. List the companies in `watchlist.txt` (one Yahoo ticker per line, e.g. `TATASTEEL.NS`). The app's *Monitor → Watchlist*
   tab prints the lines for the companies you are watching.
2. Every night the snapshot job writes a summary on the workflow run page (GitHub → Actions → Nightly data snapshot).
3. To also get it by email, add these repository secrets (GitHub → Settings → Secrets and variables → Actions):
   `ALERT_SMTP_HOST` (e.g. `smtp.gmail.com`), `ALERT_SMTP_PORT` (`465`), `ALERT_SMTP_USER` (the sending address),
   `ALERT_SMTP_PASSWORD` (for Gmail, an *app password*, not your normal password) and `ALERT_TO` (where to send it).
