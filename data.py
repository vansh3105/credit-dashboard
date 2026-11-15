"""Data layer: live Yahoo Finance (yfinance).

Pulls annual balance sheet / income statement / cash flow for each ticker and maps Yahoo's row
names to the standard columns used by metrics.py (INR crore). Results are cached per company for
6 hours (shared by every visitor of the app), so a sector only has to be fetched once.
"""
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

CR = 1e7  # 1 crore = 10 million
IST = timezone(timedelta(hours=5, minutes=30))
CACHE_TTL = 6 * 3600     # successful pulls are reused for 6 hours
FAIL_TTL = 120           # failures are remembered for 2 minutes only, so a retry is quick

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
    "cff":      ("cf", ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"]),
    "capex":    ("cf", ["Capital Expenditure"]),
    "divs":     ("cf", ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
    "netdebt":  ("cf", ["Net Issuance Payments Of Debt", "Net Long Term Debt Issuance"]),
}
NON_MONEY = {"shares"}
OPTIONAL = {"shares", "sga", "cur_debt", "retained", "ltd", "cff", "divs", "netdebt"}   # missing is tolerated


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
    period_end: str = ""                             # e.g. "Mar-2026"
    fetched_at: str = ""                             # e.g. "15 Nov 2026, 00:20 IST"


def _first_row(df, names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def fiscal_label(period_end):
    """Indian fiscal-year label for a period end. FY2026 = Apr-2025 to Mar-2026, so a March year end keeps its calendar
    year, while an October-December year end (e.g. Dec-2025) overlaps FY2026 for 9 of 12 months and is labelled FY2026.
    Without this, December year-end companies sat a year behind their March year-end peers."""
    ts = pd.Timestamp(period_end)
    return ts.year + 1 if ts.month >= 10 else ts.year


def _standardise(ticker, name, bs, is_, cf, info):
    stm = {"bs": bs, "is": is_, "cf": cf}
    cols, missing = {}, []
    for k, (s, names) in MAP.items():
        row = _first_row(stm[s], names)
        if row is None:
            missing.append(k)
            continue
        row = pd.to_numeric(row, errors="coerce")
        row.index = [fiscal_label(x) for x in row.index]
        row = row[~row.index.duplicated()]
        cols[k] = row if k in NON_MONEY else row / CR
    fin = pd.DataFrame(cols).sort_index().dropna(how="all")
    if "ebitda" not in fin and {"ebit", "da"} <= set(fin):
        fin["ebitda"] = fin["ebit"] + fin["da"]
    if "ebit" not in fin and {"pbt", "interest"} <= set(fin):
        fin["ebit"] = fin["pbt"] + fin["interest"]
    core = [c for c in ["rev", "ta", "cfo"] if c in fin]
    fin = fin.dropna(subset=core, how="any") if core else fin
    # period end date of the latest year, for the "data as of" stamp
    period_end = ""
    try:
        periods = {fiscal_label(c): pd.Timestamp(c) for c in is_.columns}
        if len(fin):
            period_end = periods[fin.index[-1]].strftime("%b-%Y")
    except Exception:
        pass
    raw = {}
    for df in (bs, is_, cf):
        if df is None or df.empty:
            continue
        for col in df.columns:
            yr = fiscal_label(col)
            raw.setdefault(yr, []).extend([float(v) for v in df[col].dropna().values if v])
    gov = {k: info.get(k) for k in ("auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk")
           if info.get(k) is not None}
    mcap = info.get("marketCap") or np.nan
    return Company(ticker, name, fin, mcap / CR if mcap == mcap else np.nan, raw, gov,
                   info.get("financialCurrency", "INR"),
                   [m for m in missing if m not in OPTIONAL] + [m for m in missing if m in OPTIONAL],
                   period_end, datetime.now(IST).strftime("%d %b %Y, %H:%M IST"))


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


# ------------------------------------------------------------------ per-company cache (shared by all visitors)
_CACHE = {}          # ticker -> (timestamp, Company | None, error | None)
_LOCK = threading.Lock()


def _fresh(entry):
    ts, comp, _ = entry
    return time.time() - ts < (CACHE_TTL if comp is not None else FAIL_TTL)


def is_cached(ticker):
    with _LOCK:
        e = _CACHE.get(ticker)
    return bool(e) and _fresh(e)


PREFER_LIVE = False      # set by clear_cache(): after "Reload data" go to Yahoo even if a fresh snapshot exists


def _snap():
    import snapshot
    return snapshot


def source():
    """Where data comes from right now: dict(mode='snapshot'|'live', built=datetime|None, age_h=float)."""
    sn = _snap()
    s = sn.get()
    return {"mode": "snapshot" if (s and sn.is_fresh(s) and not PREFER_LIVE) else "live",
            "built": sn.built_at(s), "age_h": sn.age_hours(s)}


def get_company(ticker, name):
    """Return (Company | None, error | None), using the 6-hour cache when possible.
    Order: memory cache -> fresh nightly snapshot -> live Yahoo -> any saved snapshot (marked as a saved copy)."""
    with _LOCK:
        e = _CACHE.get(ticker)
    if e and _fresh(e):
        return e[1], e[2]
    sn = _snap()
    if not PREFER_LIVE and sn.is_fresh(sn.get()):
        comp = sn.company(ticker)
        if comp is not None:
            with _LOCK:
                _CACHE[ticker] = (time.time(), comp, None)
            return comp, None
    comp, err = fetch_live(ticker, name)
    if comp is None:
        saved = sn.company(ticker)
        if saved is not None:                  # Yahoo failed: fall back to the last good copy
            saved.fetched_at = f"{saved.fetched_at} (saved copy - Yahoo unavailable)"
            comp, err = saved, None
    with _LOCK:
        _CACHE[ticker] = (time.time(), comp, err)
    return comp, err


def mark_failed(ticker, error):
    with _LOCK:
        _CACHE[ticker] = (time.time(), None, error)


def clear_cache():
    global PREFER_LIVE
    PREFER_LIVE = True
    _snap().reset()
    with _LOCK:
        _CACHE.clear()
        _QCACHE.clear()
        _PCACHE.clear()


# ------------------------------------------------------------------ early-warning data: quarterly results and prices
QMAP_IS = {"rev": ["Total Revenue", "Operating Revenue"], "ebitda": ["EBITDA", "Normalized EBITDA"],
           "ebit": ["EBIT", "Operating Income"], "ni": ["Net Income", "Net Income Common Stockholders"],
           "interest": ["Interest Expense", "Interest Expense Non Operating"], "da": ["Reconciled Depreciation"]}
QMAP_BS = {"debt": ["Total Debt"], "cur_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
           "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
           "ca": ["Current Assets"], "cl": ["Current Liabilities"], "ta": ["Total Assets"]}
_QCACHE, _PCACHE = {}, {}


def _frame(df, mapping):
    """Yahoo statement (rows = items, columns = period ends) -> DataFrame indexed by period end, Rs crore."""
    cols = {}
    for k, names in mapping.items():
        row = _first_row(df, names)
        if row is not None:
            cols[k] = pd.to_numeric(row, errors="coerce") / CR
    out = pd.DataFrame(cols)
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index)
    return out.sort_index().dropna(how="all")


def get_quarterly(ticker):
    """{'is': quarterly income statement, 'bs': interim (usually half-yearly) balance sheet}, cached 6 hours.
    Indian companies publish no quarterly cash-flow statement, so none is returned."""
    with _LOCK:
        e = _QCACHE.get(ticker)
    if e and time.time() - e[0] < CACHE_TTL:
        return e[1]
    sn = _snap()
    saved = sn.quarterly(ticker)
    if saved is not None and not PREFER_LIVE and sn.is_fresh(sn.get()):
        with _LOCK:
            _QCACHE[ticker] = (time.time(), saved)
        return saved
    import yfinance as yf
    res = {"is": pd.DataFrame(), "bs": pd.DataFrame()}
    try:
        t = yf.Ticker(ticker)
        res = {"is": _frame(t.quarterly_income_stmt, QMAP_IS), "bs": _frame(t.quarterly_balance_sheet, QMAP_BS)}
    except Exception:
        pass
    if not len(res["is"]) and saved is not None:
        res = saved
    with _LOCK:
        _QCACHE[ticker] = (time.time() if len(res["is"]) else time.time() - CACHE_TTL + FAIL_TTL, res)
    return res


def get_prices(tickers):
    """Daily adjusted close for the last 2 years, one column per ticker (one batch request), cached 6 hours."""
    key = tuple(sorted(tickers))
    with _LOCK:
        e = _PCACHE.get(key)
    if e and time.time() - e[0] < CACHE_TTL:
        return e[1]
    sn = _snap()
    saved = sn.prices(key)
    if not PREFER_LIVE and sn.is_fresh(sn.get()) and len(saved) and set(key) <= set(saved.columns):
        with _LOCK:
            _PCACHE[key] = (time.time(), saved)
        return saved
    import yfinance as yf
    try:
        px = yf.download(list(key), period="2y", auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(key[0])
        px = px.dropna(how="all")
    except Exception:
        px = pd.DataFrame()
    if px.empty and len(saved):
        px = saved
    with _LOCK:
        _PCACHE[key] = (time.time() if len(px) else time.time() - CACHE_TTL + FAIL_TTL, px)
    return px
