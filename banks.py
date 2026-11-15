"""Banks and NBFCs: their own ratios, cut-offs and score.

A lender's balance sheet is its product, so current ratio, EBITDA or debt/equity mean little. What matters is capital,
asset quality, earnings and efficiency (the CAMELS idea). Yahoo gives the statements but not bad-loan (NPA) or
capital-adequacy (CRAR) figures, so those can be entered from the company's results; credit cost is derived.

Derived from Yahoo (Rs crore; averages of opening and closing balances):
  ROA            net profit / average total assets
  ROE            net profit / average equity
  NIM (on assets) net interest income / average total assets
  Equity / assets capital proxy (not CRAR)
  Borrowings / equity  NBFC leverage (banks fund mainly with deposits, so shown for context only)
  Cost-to-income operating expenses / total income
  Credit cost (est.) (total income - operating expenses - profit before tax) / average loans
                 = pre-provision profit minus pre-tax profit: provisions and other one-offs, an estimate
  Loan growth, yield on loans, cost of funds - context, not rated
Entered by the analyst from results: Gross NPA %, Net NPA %, CRAR %.
Fallbacks: NII = interest income - interest expense when both are reported; with no loan line, loans = assets - cash
- investments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# key: (label, pillar, unit, dir, (bank green, bank red), (nbfc green, nbfc red), description)
BM = {
    "roa": ("Return on assets", "Earnings", "%", "high", (0.010, 0.005), (0.020, 0.010),
            "Net profit / average assets. The core profitability measure for a lender."),
    "roe": ("Return on equity", "Earnings", "%", "high", (0.14, 0.08), (0.14, 0.08), "Net profit / average equity."),
    "nim": ("Net interest margin (on assets)", "Earnings", "%", "high", (0.030, 0.025), (0.050, 0.030),
            "Net interest income / average total assets - the spread the lender earns."),
    "eq_assets": ("Equity / assets", "Capital", "%", "high", (0.09, 0.06), (0.15, 0.10),
                  "Book equity as a share of assets - a simple capital cushion (not the regulatory CRAR)."),
    "borr_eq": ("Borrowings / equity", "Capital", "x", "low", (None, None), (4.0, 6.0),
                "Market borrowings per rupee of equity. Rated for NBFCs; banks fund mainly with deposits."),
    "crar": ("CRAR (entered)", "Capital", "%", "high", (0.14, 0.115), (0.20, 0.15),
             "Capital to risk-weighted assets, from the results. RBI minimum: 11.5% for banks incl. buffer, 15% for NBFCs."),
    "credit_cost": ("Credit cost (estimated)", "Asset quality", "%", "low", (0.010, 0.020), (0.015, 0.030),
                    "Pre-provision profit minus pre-tax profit, over average loans - mostly loan-loss provisions."),
    "gnpa": ("Gross NPA (entered)", "Asset quality", "%", "low", (0.03, 0.06), (0.03, 0.06),
             "Loans overdue 90+ days as a share of all loans, from the results."),
    "nnpa": ("Net NPA (entered)", "Asset quality", "%", "low", (0.01, 0.03), (0.015, 0.04),
             "Bad loans net of provisions. RBI's prompt corrective action starts at 6% for banks."),
    "cost_income": ("Cost-to-income", "Efficiency", "%", "low", (0.45, 0.55), (0.40, 0.50),
                    "Operating expenses / total income. Lower means more of each rupee earned is left for losses and profit."),
    "loan_growth": ("Loan growth", "Context", "%", None, (None, None), (None, None),
                    "Year-on-year growth in loans. Very fast growth often comes before asset-quality problems."),
    "yield": ("Interest income / loans", "Context", "%", None, (None, None), (None, None),
              "Interest income / average loans (includes interest on investments, so above the true loan yield)."),
    "cost_funds": ("Cost of funds", "Context", "%", None, (None, None), (None, None), "Interest expense / average liabilities."),
    "loans_assets": ("Loans / assets", "Context", "%", None, (None, None), (None, None), "Share of the balance sheet lent out."),
}
PILLARS = ["Capital", "Asset quality", "Earnings", "Efficiency"]
WEIGHTS = {"Capital": 30, "Asset quality": 30, "Earnings": 25, "Efficiency": 15}
MANUAL = ("gnpa", "nnpa", "crar")


def _avg(s: pd.Series) -> pd.Series:
    return (s + s.shift(1)) / 2


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.where(b.abs() > 0) if isinstance(out, pd.Series) else out


def compute(fin: pd.DataFrame) -> pd.DataFrame:
    """Rows = metric keys, columns = fiscal years. The first year has no averages, so balance-based ratios use closing values."""
    d = fin.sort_index()
    c = lambda k: d[k].astype(float) if k in d else pd.Series(np.nan, index=d.index)
    ta, eq, loans, debt, tl = c("ta"), c("equity"), c("loans"), c("debt"), c("tl")
    rev, nii, ii, ie, pbt, ni = c("rev"), c("nii"), c("ii"), c("ie"), c("pbt"), c("ni")
    # Interest income minus interest expense when both are reported (Yahoo's own NII line is sometimes wrong)
    nii = (ii - ie).where(ii.notna() & ie.notna(), nii)
    # If reported interest income is under half of total income, Yahoo has filed the loan income elsewhere (e.g. IREDA):
    # interest-based ratios and credit cost would be wrong, so they are left blank.
    bad_int = ~(ii >= 0.5 * rev)
    nii = nii.where(~bad_int)
    # No loan line (some NBFCs): assets less cash and investments is close to the loan book
    est = ta - c("cash").fillna(0) - c("invest").fillna(0)
    loans = loans.where(loans.notna(), est.where(est > 0))
    opex = c("opex_op")
    alt = c("sga").fillna(0) + c("occ").fillna(0)
    opex = opex.where(opex.notna(), alt.where(alt > 0))
    av = lambda s: _avg(s).fillna(s)
    out = {
        "roa": _div(ni, av(ta)), "roe": _div(ni, av(eq)), "nim": _div(nii, av(ta)), "eq_assets": _div(eq, ta),
        "borr_eq": _div(debt, eq), "cost_income": _div(opex, rev),
        "credit_cost": _div(rev - opex - pbt, av(loans)).where(~bad_int),
        "loan_growth": loans.pct_change(fill_method=None), "yield": _div(ii, av(loans)).where(~bad_int),
        "cost_funds": _div(ie, av(tl)).where(~bad_int),
        "loans_assets": _div(loans, ta),
    }
    m = pd.DataFrame(out).T
    m.attrs["notes"] = (["Interest lines in the Yahoo statements look incomplete (interest income under half of total income), "
                         "so NIM, yield, cost of funds and credit cost are left blank."] if bool(bad_int.iloc[-1]) else [])
    if "loans" not in d or d["loans"].isna().all():
        m.attrs["notes"].append("No loan line in the statements: the loan book is estimated as assets minus cash and investments.")
    for k in MANUAL:
        m.loc[k] = np.nan
    notes = m.attrs.get("notes", [])
    m = m.loc[[k for k in BM]]
    m.attrs["notes"] = notes
    return m


def thresholds(key, kind):
    spec = BM[key]
    return spec[4] if kind == "bank" else spec[5]


def rag(key, v, kind):
    g, r = thresholds(key, kind)
    if v is None or pd.isna(v) or g is None or BM[key][3] is None:
        return None
    if BM[key][3] == "high":
        return "G" if v >= g else ("R" if v < r else "A")
    return "G" if v <= g else ("R" if v > r else "A")


PTS = {"G": 100.0, "A": 50.0, "R": 0.0}


def latest(m: pd.DataFrame) -> pd.Series:
    return m.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)


def pillar_scores(vals: pd.Series, kind: str) -> dict:
    out = {}
    for p in PILLARS:
        pts = [PTS[r] for k in BM if BM[k][1] == p for r in [rag(k, vals.get(k), kind)] if r]
        out[p] = float(np.mean(pts)) if pts else np.nan
    return out


def score(pillars: dict, weights=None) -> float:
    w = weights or WEIGHTS
    num = sum(pillars[p] * wt for p, wt in w.items() if not pd.isna(pillars.get(p, np.nan)))
    den = sum(wt for p, wt in w.items() if not pd.isna(pillars.get(p, np.nan)))
    return num / den if den else np.nan


def signal(vals: pd.Series, kind: str) -> str:
    reds = [k for k in BM if rag(k, vals.get(k), kind) == "R"]
    core = {"roa", "eq_assets", "crar", "credit_cost", "gnpa", "nnpa"}
    if len(reds) >= 3 or len(core & set(reds)) >= 2:
        return "ELEVATED CONCERN"
    if reds:
        return "WATCH"
    return "COMFORTABLE"


def fmt(key, v):
    if v is None or pd.isna(v):
        return "n/a"
    return f"{v:.2f}x" if BM[key][2] == "x" else f"{v * 100:.2f}%"


def with_manual(vals: pd.Series, manual: dict) -> pd.Series:
    """Add the analyst's NPA / CRAR entries (given in %, stored as fractions)."""
    v = vals.copy()
    for k in MANUAL:
        x = (manual or {}).get(k)
        if x is not None and not pd.isna(x):
            v[k] = x / 100
    return v
