"""Data layer: live Yahoo Finance (yfinance).

Live mode pulls annual balance sheet / income statement / cash flow for each ticker and
maps Yahoo's row names to the standard columns used by metrics.py (INR crore).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CR = 1e7  # 1 crore = 10 million

# standard column -> (statement, [Yahoo row-name candidates in priority order])
MAP = {
    "rev":      ("is", ["Total Revenue", "Operating Revenue"]),
    "cogs":     ("is", ["Cost Of Revenue", "Reconciled Cost Of Revenue"]),
    "ebit":     ("is", ["EBIT", "Operating Income"]),
    "ebitda":   ("is", ["EBITDA", "Normalized EBITDA"]),
    "ni":       ("is", ["Net Income", "Net Income Common Stockholders"]),
    "pbt":      ("is", ["Pretax Income"]),
    "interest": ("is", ["Interest Expense", "Interest Expense Non Operating"]),
    "tax":      ("is", ["Tax Provision"]),
    "da":       ("is", ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"]),
    "sga":      ("is", ["Selling General And Administration", "Selling And Marketing Expense"]),
    "ta":       ("bs", ["Total Assets"]),
    "ca":       ("bs", ["Current Assets"]),
    "cl":       ("bs", ["Current Liabilities"]),
    "cash":     ("bs", ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]),
    "recv":     ("bs", ["Accounts Receivable", "Receivables"]),
    "inv":      ("bs", ["Inventory"]),
    "pay":      ("bs", ["Accounts Payable", "Payables"]),
    "debt":     ("bs", ["Total Debt"]),
    "ltd":      ("bs", ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]),
    "cur_debt": ("bs", ["Current Debt", "Current Debt And Capital Lease Obligation"]),
    "equity":   ("bs", ["Stockholders Equity", "Common Stock Equity"]),
    "retained": ("bs", ["Retained Earnings"]),
    "tl":       ("bs", ["Total Liabilities Net Minority Interest"]),
    "ppe":      ("bs", ["Net PPE"]),
    "shares":   ("bs", ["Share Issued", "Ordinary Shares Number"]),
    "cfo":      ("cf", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    "cfi":      ("cf", ["Investing Cash Flow", "Cash Flow From Continuing Investing Activities"]),
    "capex":    ("cf", ["Capital Expenditure"]),
}
NON_MONEY = {"shares"}


@dataclass
class Company:
    ticker: str
    name: str
    fin: pd.DataFrame            # standard columns, index = fiscal year, INR crore
    mcap_cr: float = np.nan
    raw_values: dict = field(default_factory=dict)   # year -> list of raw statement numbers (for Benford)
    gov: dict = field(default_factory=dict)          # Yahoo/ISS governance risk scores if present
    currency: str = "INR"
    missing: list = field(default_factory=list)      # standard fields Yahoo did not provide


def _first_row(df, names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _standardise(ticker, name, bs, is_, cf, info):
    stm = {"bs": bs, "is": is_, "cf": cf}
    cols = {}
    missing = []
    for k, (s, names) in MAP.items():
        row = _first_row(stm[s], names)
        if row is None:
            missing.append(k)
            continue
        row = pd.to_numeric(row, errors="coerce")
        row.index = pd.to_datetime(row.index).year          # fiscal-year label = calendar year of period end
        row = row[~row.index.duplicated()]
        cols[k] = row if k in NON_MONEY else row / CR
    fin = pd.DataFrame(cols).sort_index().dropna(how="all")
    # EBITDA / EBIT fallbacks
    if "ebitda" not in fin and {"ebit", "da"} <= set(fin):
        fin["ebitda"] = fin["ebit"] + fin["da"]
    if "ebit" not in fin and {"pbt", "interest"} <= set(fin):
        fin["ebit"] = fin["pbt"] + fin["interest"]
    # keep only years with core data
    core = [c for c in ["rev", "ta", "cfo"] if c in fin]
    fin = fin.dropna(subset=core, how="any") if core else fin
    # raw numbers for Benford
    raw = {}
    for df in (bs, is_, cf):
        if df is None or df.empty:
            continue
        for col in df.columns:
            yr = pd.Timestamp(col).year
            raw.setdefault(yr, []).extend([float(v) for v in df[col].dropna().values if v])
    gov = {k: info.get(k) for k in ("auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk")
           if info.get(k) is not None}
    mcap = (info.get("marketCap") or np.nan)
    return Company(ticker, name, fin, mcap / CR if mcap == mcap else np.nan, raw, gov,
                   info.get("financialCurrency", "INR"),
                   [m for m in missing if m not in ("shares", "sga", "cur_debt", "retained", "ltd")] + [m for m in missing if m in ("sga", "cur_debt", "ltd", "retained")])


def fetch_live(ticker, name, retries=2):
    import yfinance as yf
    last = None
    for a in range(retries):
        try:
            t = yf.Ticker(ticker)
            bs, is_, cf = t.balance_sheet, t.financials, t.cashflow
            if bs is None or bs.empty or is_ is None or is_.empty:
                raise ValueError("Yahoo returned no annual statements")
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}
            return _standardise(ticker, name, bs, is_, cf, info), None
        except Exception as e:                      # network, rate limit, unknown ticker
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(1.0)
    return None, last


def load_sector(tickers: dict):
    """Fetch a whole sector. Returns (dict ticker->Company, dict ticker->error).
    Pulls run one at a time (avoids Yahoo rate-limits) and stop early after 3 consecutive
    failures so an unreachable network does not make the user wait minutes."""
    ok, err = {}, {}
    streak = 0
    items = list(tickers.items())
    for i, (t, n) in enumerate(items):
        c, e = fetch_live(t, n)
        if c is None or c.fin.empty:
            err[t] = e or "no usable data"
            streak += 1
            if streak >= 3 and not ok:
                for t2, _ in items[i + 1:]:
                    err[t2] = "skipped - Yahoo Finance unreachable (first 3 requests failed)"
                break
        else:
            ok[t] = c
            streak = 0
    return ok, err
