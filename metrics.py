"""Credit-risk metric engine. Every formula comes from the FSAFA course decks
(session in brackets). All money inputs are in INR crore.

Input: DataFrame `d`, index = fiscal year (ascending), columns (standard names):
 rev cogs ebit ebitda ni pbt interest tax da sga ta ca cl cash recv inv pay debt ltd
 cur_debt equity retained tl ppe cfo cfi capex shares
Output: DataFrame, index = metric key, columns = fiscal years.
"""
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ registry
# key: (label, group, unit, direction, green, red, formula, source)
#  direction: "high" = higher is better, "low" = lower is better, None = informational
#  green/red thresholds: value at/beyond green => Green; at/beyond red => Red; between => Amber
M = {}
def _r(key, label, group, unit, direction, green, red, formula, src):
    M[key] = dict(label=label, group=group, unit=unit, dir=direction, green=green, red=red, formula=formula, src=src)

# Liquidity  (Sessions 5-6)
_r("current_ratio", "Current ratio", "Liquidity", "x", "high", 1.5, 1.0, "CA / CL", "S5-6 (starting point only)")
_r("quick_ratio", "Quick ratio", "Liquidity", "x", "high", 1.0, 0.7, "(CA - Inventory) / CL", "S5-6")
_r("cash_ratio", "Cash ratio", "Liquidity", "x", "high", 0.3, 0.1, "Cash / CL", "S5-6")
_r("wc_days", "Working-capital days", "Liquidity", "days", "low", 45, 90, "(CA - CL) / Sales x 365", "S5-6")
_r("ccc", "Cash conversion cycle", "Liquidity", "days", None, None, None, "DSO + DIO - DPO", "S5-6")
_r("ocf_ratio", "Operating cash-flow ratio", "Liquidity", "x", "high", 0.4, 0.2, "CFO / CL", "S5-6")
_r("cce", "Cash conversion efficiency", "Liquidity", "%", "high", 0.8, 0.4, "CFO / EBITDA", "S5-6 (<40% major concern, 80%+ strong)")
_r("dir_days", "Defensive interval (days)", "Liquidity", "days", "high", 150, 60, "(Cash + Receivables) / (Cash opex / 365)", "S5-6")
_r("runway_m", "Cash runway (months, only if CFO<0)", "Liquidity", "months", "high", 24, 12, "Cash / monthly burn", "S5-6")
# Solvency / debt service  (Sessions 3-4, 7-8)
_r("de", "Debt / Equity", "Solvency", "x", "low", 1.0, 2.0, "Total debt / Equity", "S3-4, S7-8")
_r("debt_ratio", "Debt ratio (Debt / Assets)", "Solvency", "x", "low", 0.4, 0.7, "Total debt / Total assets", "S3-4")
_r("equity_ratio", "Equity ratio", "Solvency", "x", "high", 0.4, 0.2, "Equity / Total assets", "S3-4")
_r("icr", "Interest coverage", "Solvency", "x", "high", 3.0, 1.5, "EBIT / Interest", "S3-4 (1.5x = highly leveraged)")
_r("dscr", "DSCR (proxy)", "Solvency", "x", "high", 1.5, 1.0, "(CFO + Interest) / (Interest + current debt)", "S7-8 (<1 = cash insufficient)")
_r("nd_ebitda", "Net debt / EBITDA", "Solvency", "x", "low", 2.0, 4.0, "(Total debt - Cash) / EBITDA", "extension")
_r("fin_lev", "Financial leverage (Assets/Equity)", "Solvency", "x", None, None, None, "Avg assets / Avg equity", "S1-2 DuPont")
_r("roe", "ROE", "Solvency", "%", None, None, None, "NI / Avg equity", "S1-2 DuPont")
_r("roa", "ROA", "Solvency", "%", None, None, None, "NI / Avg assets", "S1-2")
# Earnings quality (Sessions 5-8)
_r("sloan", "Sloan ratio (accruals / avg assets)", "Earnings quality", "%", "low", 0.05, 0.10, "(NI - CFO) / Avg total assets", "S7-8 (0-5% low risk, >20% very high)")
_r("bs_accrual", "Balance-sheet accrual ratio", "Earnings quality", "%", "low", 0.05, 0.15, "Change in NOA / Avg NOA", "S7-8")
_r("cf_accrual", "Cash-flow accrual ratio", "Earnings quality", "%", "low", 0.05, 0.15, "(NI - CFO) / Avg NOA", "S7-8")
_r("cfo_ni", "CFO / Net income", "Earnings quality", "x", "high", 1.0, 0.5, "CFO / NI", "S9-10 (Piotroski: CFO > NI)")
_r("ffo_tl", "FFO / Total liabilities", "Earnings quality", "%", "high", 0.25, 0.10, "(NI + D&A) / TL", "S9-10")
_r("fcf", "Free cash flow (Rs cr)", "Earnings quality", "Rs cr", "high", 0, 0, "CFO - Capex", "S5-6")
# Returns (Sessions 7-8)
_r("roce", "ROCE", "Returns", "%", None, None, None, "EBIT / (Assets - CL)", "S7-8 (credit analyst's metric)")
_r("roic", "ROIC", "Returns", "%", None, None, None, "NOPAT / Avg invested capital", "S7-8")
_r("roic_spread", "ROIC - cost of debt", "Returns", "%", "high", 0.03, 0.0, "ROIC - Interest / Avg debt", "S7-8 (ROIIC < debt cost = warning)")
_r("roiic", "ROIIC", "Returns", "%", None, None, None, "Change in NOPAT / Change in invested capital", "S7-8")
# Distress & manipulation scores (Sessions 9-12)
_r("altman_z", "Altman Z (manufacturing, uses market cap)", "Scores", "score", "high", 2.99, 1.81, "1.2X1+1.4X2+3.3X3+0.6X4+1.0X5", "S7-8: >2.99 safe, <1.81 distress")
_r("altman_zp", "Altman Z' (private mfg)", "Scores", "score", "high", 2.90, 1.23, "0.717X1+0.847X2+3.107X3+0.42X4+0.998X5", "S7-8")
_r("altman_z2", "Altman Z'' (non-mfg / emerging)", "Scores", "score", "high", 2.60, 1.10, "6.56X1+3.26X2+6.72X3+1.05X4", "S7-8")
_r("ohlson_p", "Ohlson O: distress probability", "Scores", "%", "low", 0.10, 0.50, "e^T / (1 + e^T)", "S9-10 (>50% = default within 2 yrs)")
_r("zmijewski", "Zmijewski X-score", "Scores", "score", "low", -2.0, 0.0, "-4.336-4.513 NI/TA+5.679 TL/TA+0.004 CA/CL", "S11-12 (higher = riskier)")
_r("springate", "Springate S-score", "Scores", "score", "high", 0.862, 0.862, "1.03A+3.07B+0.66C+0.4D", "S11-12 (>0.862 stable)")
_r("piotroski", "Piotroski F-score (0-9)", "Scores", "pts", "high", 7, 3, "9 binary tests", "S7-8")
_r("beneish", "Beneish M-score", "Scores", "score", "low", -2.22, -2.22, "8-variable model", "S7-8 (> -2.22 = likely manipulator)")
# Market context (Sessions 15-16)
_r("pe", "P/E", "Market", "x", None, None, None, "Market cap / NI", "S15-16")
_r("p_ocf", "P/OCF", "Market", "x", None, None, None, "Market cap / CFO", "S15-16 (P/OCF > P/E = profits not cash)")

GROUPS = ["Liquidity", "Solvency", "Earnings quality", "Returns", "Scores", "Market"]

# Plain-English meaning of each metric (shown as tooltips / guide in the app)
DESC = {
    "current_ratio": "Short-term assets for every rupee due within a year. Above 1.5 looks comfortable, but stuck receivables or slow stock can hide weakness.",
    "quick_ratio": "Like the current ratio but ignores inventory, so it tests near-cash assets against short-term dues.",
    "cash_ratio": "Cash alone against short-term dues - the most conservative liquidity test.",
    "wc_days": "Days of sales tied up in working capital. Fewer is better; a rising number means cash is getting stuck.",
    "ccc": "Days from paying suppliers to collecting from customers. Shorter is better; negative means customers and suppliers fund the business.",
    "ocf_ratio": "Operating cash generated per rupee of short-term dues.",
    "cce": "Share of EBITDA that actually turns into operating cash. Under 40% is a major concern.",
    "dir_days": "Days the firm can pay its cash costs from cash and receivables if sales stopped.",
    "runway_m": "If operations are burning cash, the months until the cash balance runs out.",
    "de": "Debt for every rupee of equity: how much lenders fund versus owners.",
    "debt_ratio": "Share of total assets funded by debt.",
    "equity_ratio": "Share of total assets funded by owners - the loss cushion for lenders.",
    "icr": "How many times operating profit (EBIT) covers interest. Below 1.5x is thin.",
    "dscr": "Cash available for debt service divided by interest plus principal due. Below 1.0 means cash does not cover repayments. Proxy built from public data.",
    "nd_ebitda": "Years of EBITDA needed to repay net debt. Above about 4x is usually stretched. (Advanced metric)",
    "fin_lev": "Rupees of assets supported by each rupee of equity; higher means more gearing.",
    "roe": "Profit earned on owners' money.",
    "roa": "Profit earned on all assets.",
    "sloan": "Share of profit that is non-cash accruals, relative to assets. High values mean lower earnings quality.",
    "bs_accrual": "Growth in net operating assets (receivables, stock) relative to their size - where did the profits go?",
    "cf_accrual": "Profit not backed by operating cash, relative to net operating assets - why didn't profit become cash?",
    "cfo_ni": "Operating cash per rupee of profit; below 1 means profits are not fully cash-backed.",
    "ffo_tl": "Funds from operations per rupee of total liabilities.",
    "fcf": "Operating cash left after capital spending.",
    "roce": "Operating profit relative to capital employed.",
    "roic": "After-tax operating profit relative to capital provided by lenders and owners.",
    "roic_spread": "ROIC minus the interest rate paid. Negative means growth is not covering the cost of debt.",
    "roiic": "Return on the additional capital invested this year.",
    "altman_z": "Bankruptcy-risk score using market value of equity. Below 1.81 is distress, above 2.99 is safe.",
    "altman_zp": "Altman score for private manufacturers (book value of equity).",
    "altman_z2": "Altman score for non-manufacturers and emerging-market firms. Below 1.10 is distress, above 2.60 is safe.",
    "ohlson_p": "Estimated probability of distress from nine ratios. Above 50% signals likely default within two years.",
    "zmijewski": "Bankruptcy-risk score from profitability, leverage and liquidity. Higher means riskier.",
    "springate": "Financial-stress score. Below 0.862 signals stress.",
    "piotroski": "Count of nine financial-strength tests passed. 7 or more is strong, 3 or less is weak.",
    "beneish": "Earnings-manipulation score. Above -2.22 suggests a likely manipulator.",
    "pe": "Price paid for each rupee of profit.",
    "p_ocf": "Price paid for each rupee of operating cash flow. Higher than P/E means profits are not converting to cash.",
}
for _k, _v in M.items():
    _v["desc"] = DESC.get(_k, "")

# Standard input fields each metric needs (used to explain why a metric is n/a)
NEEDS = {
    "current_ratio": ["ca", "cl"], "quick_ratio": ["ca", "inv", "cl"], "cash_ratio": ["cash", "cl"],
    "wc_days": ["ca", "cl", "rev"], "ccc": ["recv", "inv", "pay", "rev", "cogs"], "ocf_ratio": ["cfo", "cl"],
    "cce": ["cfo", "ebitda"], "dir_days": ["cash", "recv", "rev", "ebitda"], "runway_m": ["cash", "cfo"],
    "de": ["debt", "equity"], "debt_ratio": ["debt", "ta"], "equity_ratio": ["equity", "ta"], "icr": ["ebit", "interest"],
    "dscr": ["cfo", "interest", "cur_debt"], "nd_ebitda": ["debt", "cash", "ebitda"], "fin_lev": ["ta", "equity"],
    "roe": ["ni", "equity"], "roa": ["ni", "ta"], "sloan": ["ni", "cfo", "ta"], "bs_accrual": ["ta", "cash", "tl", "debt"],
    "cf_accrual": ["ni", "cfo", "ta", "cash", "tl", "debt"], "cfo_ni": ["cfo", "ni"], "ffo_tl": ["ni", "da", "tl"],
    "fcf": ["cfo", "capex"], "roce": ["ebit", "ta", "cl"], "roic": ["ebit", "equity", "debt", "cash"],
    "roic_spread": ["ebit", "equity", "debt", "interest"], "roiic": ["ebit", "equity", "debt", "cash"],
    "altman_z": ["ca", "cl", "ta", "retained", "ebit", "tl", "rev"], "altman_zp": ["ca", "cl", "ta", "retained", "ebit", "equity", "tl", "rev"],
    "altman_z2": ["ca", "cl", "ta", "retained", "ebit", "equity", "tl"], "ohlson_p": ["ta", "tl", "ca", "cl", "ni", "da"],
    "zmijewski": ["ni", "ta", "tl", "ca", "cl"], "springate": ["ca", "cl", "ta", "ebit", "pbt", "rev"],
    "piotroski": ["ni", "cfo", "ta", "ca", "cl", "rev", "cogs"],
    "beneish": ["recv", "rev", "cogs", "ca", "ppe", "ta", "da", "sga", "cl", "ltd", "ni", "cfo"],
    "pe": ["ni"], "p_ocf": ["cfo"],
}
FIELD_NAME = {"ca": "current assets", "cl": "current liabilities", "inv": "inventory", "recv": "receivables", "pay": "payables",
              "cogs": "cost of revenue", "sga": "SG&A", "da": "depreciation", "ppe": "net PP&E", "ltd": "long-term debt",
              "cur_debt": "current debt", "retained": "retained earnings", "capex": "capex", "ebitda": "EBITDA", "ebit": "EBIT",
              "interest": "interest expense", "cash": "cash", "debt": "total debt", "equity": "equity", "tl": "total liabilities",
              "ta": "total assets", "rev": "revenue", "ni": "net income", "pbt": "pre-tax income", "cfo": "operating cash flow"}


def missing_inputs(key, fin):
    """Names of the input fields a metric needs that Yahoo did not supply for this company."""
    out = []
    for n in NEEDS.get(key, []):
        if n not in fin.columns or fin[n].isna().all():
            out.append(FIELD_NAME.get(n, n))
    return out



# ------------------------------------------------------------------ helpers
def div(a, b):
    b = b.where(b.abs() > 1e-12)
    return a / b

def avg(s):
    """Two-point average (current + prior year); first year falls back to year-end value."""
    return ((s + s.shift(1)) / 2).fillna(s)

def rag(key, v):
    """Return 'G','A','R' or None for a metric value."""
    m = M[key]
    if v is None or pd.isna(v) or m["dir"] is None:
        return None
    g, r = m["green"], m["red"]
    if key == "springate":
        return "G" if v > 0.862 else "R"
    if key == "beneish":
        return "G" if v < -2.22 else "R"
    if key == "fcf":
        return "G" if v > 0 else "R"
    if m["dir"] == "high":
        return "G" if v >= g else ("R" if v < r else "A")
    return "G" if v <= g else ("R" if v > r else "A")


# ------------------------------------------------------------------ scores
def ohlson_T(ta, tl, wc, ca, cl, ni, ffo, ni_prev, gnp=100.0):
    """Ohlson O-score exactly as given in Sessions 9-10. `ta` in same unit as gnp index."""
    X = 1.0 if tl > ta else 0.0
    Y = 1.0 if (ni < 0 and ni_prev < 0) else 0.0
    chg = (ni - ni_prev) / (abs(ni) + abs(ni_prev)) if (abs(ni) + abs(ni_prev)) else 0.0
    return (-1.32 - 0.407 * np.log(ta / gnp) + 6.03 * tl / ta - 1.43 * wc / ta + 0.0757 * cl / ca
            - 1.72 * X - 2.37 * ni / ta - 1.83 * ffo / tl + 0.285 * Y - 0.521 * chg)

def logistic(t):
    return 1.0 / (1.0 + np.exp(-t))

def beneish_m(idx):
    """idx: dict with DSRI GMI AQI SGI DEPI SGAI LVGI TATA. Missing -> neutral 1 (TATA -> NaN)."""
    g = lambda k: 1.0 if (idx.get(k) is None or pd.isna(idx.get(k))) else idx[k]
    if idx.get("TATA") is None or pd.isna(idx.get("TATA")):
        return np.nan
    return (-4.84 + 0.92 * g("DSRI") + 0.528 * g("GMI") + 0.404 * g("AQI") + 0.892 * g("SGI")
            + 0.115 * g("DEPI") - 0.172 * g("SGAI") + 4.679 * idx["TATA"] - 0.327 * g("LVGI"))


# ------------------------------------------------------------------ main engine
def compute(d: pd.DataFrame, mcap_cr: float = np.nan, gnp: float = 100.0) -> pd.DataFrame:
    d = d.sort_index().copy()
    out = {}
    c = lambda k: d[k] if k in d else pd.Series(np.nan, index=d.index)
    rev, cogs, ebit, ebitda, ni, pbt, intr, tax, da, sga = [c(k) for k in
        "rev cogs ebit ebitda ni pbt interest tax da sga".split()]
    ta, ca, cl, cash, recv, inv, pay, debt, ltd, cur_debt, eq, re_, tl, ppe, cfo, cfi, capex = [c(k) for k in
        "ta ca cl cash recv inv pay debt ltd cur_debt equity retained tl ppe cfo cfi capex".split()]
    shares = c("shares")
    ta_avg, eq_avg = avg(ta), avg(eq)
    wc = ca - cl

    # Liquidity
    out["current_ratio"] = div(ca, cl)
    out["quick_ratio"] = div(ca - inv, cl)
    out["cash_ratio"] = div(cash, cl)
    out["wc_days"] = div(wc, rev) * 365
    dso, dio, dpo = div(recv, rev) * 365, div(inv, cogs) * 365, div(pay, cogs) * 365
    out["ccc"] = dso + dio - dpo
    out["ocf_ratio"] = div(cfo, cl)
    out["cce"] = div(cfo, ebitda)
    cash_opex = rev - ebitda
    out["dir_days"] = div(cash + recv, cash_opex / 365)
    burn = (-cfo / 12).where(cfo < 0)
    out["runway_m"] = div(cash, burn)
    # Solvency
    out["de"] = div(debt, eq)
    out["debt_ratio"] = div(debt, ta)
    out["equity_ratio"] = div(eq, ta)
    out["icr"] = div(ebit, intr)
    out["dscr"] = div(cfo + intr, intr + cur_debt.fillna(0))
    out["nd_ebitda"] = div(debt - cash, ebitda)
    out["fin_lev"] = div(ta_avg, eq_avg)
    out["roe"] = div(ni, eq_avg)
    out["roa"] = div(ni, ta_avg)
    # Earnings quality
    out["sloan"] = div(ni - cfo, ta_avg)
    noa = (ta - cash) - (tl - debt)
    noa_avg = avg(noa)
    out["bs_accrual"] = div(noa - noa.shift(1), noa_avg)
    out["cf_accrual"] = div(ni - cfo, noa_avg)
    out["cfo_ni"] = div(cfo, ni.where(ni > 0))
    out["ffo_tl"] = div(ni + da, tl)
    out["fcf"] = cfo + capex            # capex stored negative
    # Returns
    out["roce"] = div(ebit, ta - cl)
    tr = div(tax, pbt).clip(0, 0.40).fillna(0.25)
    nopat = ebit * (1 - tr)
    ic = eq + debt - cash
    out["roic"] = div(nopat, avg(ic))
    out["roic_spread"] = out["roic"] - div(intr, avg(debt))
    out["roiic"] = div(nopat - nopat.shift(1), ic - ic.shift(1))
    # Altman
    x1, x2, x3, x4b, x5 = div(wc, ta), div(re_, ta), div(ebit, ta), div(eq, tl), div(rev, ta)
    out["altman_z2"] = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4b
    out["altman_zp"] = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4b + 0.998 * x5
    z = pd.Series(np.nan, index=d.index)       # original Z needs market value -> latest year only
    if not pd.isna(mcap_cr) and len(d):
        i = d.index[-1]
        z.loc[i] = 1.2 * x1[i] + 1.4 * x2[i] + 3.3 * x3[i] + 0.6 * (mcap_cr / tl[i]) + 1.0 * x5[i]
    out["altman_z"] = z
    # Ohlson
    op = pd.Series(np.nan, index=d.index)
    for i, yr in enumerate(d.index):
        try:
            T = ohlson_T(ta[yr], tl[yr], wc[yr], ca[yr], cl[yr], ni[yr], (ni + da)[yr],
                         ni.iloc[i - 1] if i > 0 else ni[yr], gnp)
            op[yr] = logistic(T)
        except Exception:
            pass
    out["ohlson_p"] = op
    out["zmijewski"] = -4.336 - 4.513 * div(ni, ta) + 5.679 * div(tl, ta) + 0.004 * div(ca, cl)
    out["springate"] = 1.03 * x1 + 3.07 * x3 + 0.66 * div(pbt, cl) + 0.4 * x5
    # Piotroski
    f = pd.Series(np.nan, index=d.index)
    gm, ato = div(rev - cogs, rev), div(rev, ta_avg)
    for i, yr in enumerate(d.index):
        if i == 0:
            continue
        p = d.index[i - 1]
        tests = [ni[yr] > 0, div(ni, ta_avg)[yr] > 0, cfo[yr] > 0, cfo[yr] > ni[yr],
                 (ltd[yr] < ltd[p]) if not (pd.isna(ltd[yr]) or pd.isna(ltd[p])) else None,
                 out["current_ratio"][yr] > out["current_ratio"][p],
                 (shares[yr] <= shares[p]) if not (pd.isna(shares[yr]) or pd.isna(shares[p])) else None,
                 gm[yr] > gm[p], ato[yr] > ato[p]]
        ok = [t for t in tests if t is not None and not pd.isna(t)]
        f[yr] = float(sum(bool(t) for t in ok)) if ok else np.nan
    out["piotroski"] = f
    # Beneish
    bm = pd.Series(np.nan, index=d.index)
    for i, yr in enumerate(d.index):
        if i == 0:
            continue
        p = d.index[i - 1]
        idx = dict(
            DSRI=div(recv, rev)[yr] / div(recv, rev)[p],
            GMI=gm[p] / gm[yr],
            AQI=(1 - (ca[yr] + ppe[yr]) / ta[yr]) / (1 - (ca[p] + ppe[p]) / ta[p]),
            SGI=rev[yr] / rev[p],
            DEPI=(da[p] / (ppe[p] + da[p])) / (da[yr] / (ppe[yr] + da[yr])),
            SGAI=(sga[yr] / rev[yr]) / (sga[p] / rev[p]),
            LVGI=((cl[yr] + ltd.fillna(0)[yr]) / ta[yr]) / ((cl[p] + ltd.fillna(0)[p]) / ta[p]),
            TATA=(ni[yr] - cfo[yr]) / ta[yr])
        bm[yr] = beneish_m(idx)
    out["beneish"] = bm
    # Market
    pe = pd.Series(np.nan, index=d.index); pocf = pe.copy()
    if not pd.isna(mcap_cr) and len(d):
        i = d.index[-1]
        pe[i] = mcap_cr / ni[i] if ni[i] > 0 else np.nan
        pocf[i] = mcap_cr / cfo[i] if cfo[i] > 0 else np.nan
    out["pe"], out["p_ocf"] = pe, pocf
    res = pd.DataFrame(out).T.replace([np.inf, -np.inf], np.nan)
    return res.reindex([k for k in M if k in res.index])


# ------------------------------------------------------------------ Benford (Sessions 9-10)
def benford(values):
    """KS = max |cumulative(actual) - cumulative(expected)|; cut-off = 1.36 / sqrt(N)."""
    v = np.abs(np.asarray([x for x in values if x is not None and not pd.isna(x) and x != 0], dtype=float))
    v = v[v > 0]
    n = len(v)
    if n < 30:
        return dict(n=n, ks=np.nan, cutoff=np.nan, flag=None, actual=None, expected=None)
    first = np.array([int(f"{x:.15e}"[0]) for x in v])
    actual = np.array([(first == k).mean() for k in range(1, 10)])
    expected = np.log10(1 + 1 / np.arange(1, 10))
    ks = float(np.max(np.abs(np.cumsum(actual) - np.cumsum(expected))))
    cutoff = 1.36 / np.sqrt(n)
    return dict(n=n, ks=ks, cutoff=cutoff, flag=bool(ks > cutoff), actual=actual, expected=expected)


# ------------------------------------------------------------------ red flags (Sessions 1-2, 5-8)
def red_flags(d: pd.DataFrame, m: pd.DataFrame):
    """List of (severity, text) for the latest year, using standard red-flag rules."""
    flags = []
    if len(d) < 2:
        return flags
    y, p = d.index[-1], d.index[-2]
    g = lambda k: (d[k][y] / d[k][p] - 1) if (k in d and d[k][p] not in (0, np.nan) and not pd.isna(d[k][p]) and not pd.isna(d[k][y])) else np.nan
    rg, rec, inv, dbt, nig, cfg = g("rev"), g("recv"), g("inv"), g("debt"), g("ni"), g("cfo")
    if d["cfo"][y] < 0:
        flags.append(("R", f"Operating cash flow is negative (Rs {d['cfo'][y]:,.0f} cr) - cash pays debt, not EBITDA."))
    if d["ni"][y] > 0 and not pd.isna(cfg) and cfg < 0 and not pd.isna(nig) and nig > 0:
        flags.append(("R", f"Profit up {nig:.0%} but CFO down {abs(cfg):.0%} - a classic 'profit up, cash down' red flag."))
    if not pd.isna(rec) and not pd.isna(rg) and rec > rg + 0.10:
        flags.append(("A", f"Receivables grew {rec:.0%} vs sales {rg:.0%} - possible aggressive revenue recognition / collection stress."))
    if not pd.isna(inv) and not pd.isna(rg) and inv > rg + 0.10:
        flags.append(("A", f"Inventory grew {inv:.0%} vs sales {rg:.0%} - inventory build-up / write-down risk."))
    if not pd.isna(dbt) and not pd.isna(rg) and dbt > rg + 0.15:
        flags.append(("A", f"Debt grew {dbt:.0%} vs sales {rg:.0%} - excess debt growth."))
    cce = m.loc["cce", y] if "cce" in m.index else np.nan
    if not pd.isna(cce) and cce < 0.4:
        flags.append(("R", f"Only {cce:.0%} of EBITDA converts into operating cash (<40% is a major concern)."))
    ndx = m.loc["nd_ebitda", y] if "nd_ebitda" in m.index else np.nan
    if not pd.isna(ndx) and ndx > 4:
        flags.append(("A", f"Net debt is {ndx:.1f}x EBITDA - stretched leverage (above about 4x)."))
    icr = m.loc["icr", y]
    if not pd.isna(icr) and icr < 1.5:
        flags.append(("R", f"Interest coverage is {icr:.1f}x - EBIT barely covers interest."))
    return flags


# ====================================================================== ADVANCED EXTENSIONS
# ---- percentile rank within sector ------------------------------------------------------
def percentile(key, value, peer_values):
    """Share (0-100) of sector values that `value` beats; higher is always better. NaN if < 3 values."""
    d = M[key]["dir"]
    vals = [v for v in peer_values if v is not None and not pd.isna(v)]
    if d is None or pd.isna(value) or len(vals) < 3:
        return np.nan
    beat = sum(1.0 for v in vals if (value > v if d == "high" else value < v)) + 0.5 * sum(1.0 for v in vals if v == value)
    return 100.0 * beat / len(vals)


# ---- indicative scorecard ---------------------------------------------------------------
PILLARS = {"Liquidity": ["Liquidity"], "Solvency & coverage": ["Solvency"], "Earnings quality": ["Earnings quality"],
           "Returns": ["Returns"], "Distress & forensic": ["Scores"]}
RAG_PTS = {"G": 100.0, "A": 50.0, "R": 0.0}
DEFAULT_WEIGHTS = {"Liquidity": 20, "Solvency & coverage": 25, "Earnings quality": 20, "Returns": 10,
                   "Distress & forensic": 20, "Governance": 5}
BANDS = [(80, "Low risk"), (65, "Moderate"), (50, "Elevated"), (35, "High"), (0, "Very high")]


def governance_score(gov):
    """Yahoo/ISS risk scores run 1 (low risk) to 10 (high risk); map to 100..0."""
    vals = [v for v in (gov or {}).values() if v is not None and not pd.isna(v)]
    return float(100 * (10 - np.mean(vals)) / 9) if vals else np.nan


def pillar_scores(vals, gov=None):
    """vals: Series metric->value. Each pillar = average of 100/50/0 for green/amber/red metrics in it."""
    out = {}
    for p, groups in PILLARS.items():
        pts = [RAG_PTS[r] for k, m in M.items() if m["group"] in groups for r in [rag(k, vals.get(k))] if r]
        out[p] = float(np.mean(pts)) if pts else np.nan
    out["Governance"] = governance_score(gov)
    return out


def overall_score(pillars, weights=None):
    w = weights or DEFAULT_WEIGHTS
    num = den = 0.0
    for p, wt in w.items():
        s_ = pillars.get(p)
        if s_ is not None and not pd.isna(s_) and wt > 0:
            num += s_ * wt; den += wt
    return num / den if den else np.nan


def band(score):
    if pd.isna(score):
        return "n/a"
    for lo, name in BANDS:
        if score >= lo:
            return name
    return "Very high"


def score_by_year(m_df, weights=None):
    """Overall indicative score for every year column (governance excluded - it has no history)."""
    w = {k: v for k, v in (weights or DEFAULT_WEIGHTS).items() if k != "Governance"}
    return pd.Series({y: overall_score(pillar_scores(m_df[y]), w) for y in m_df.columns})


# ---- stress test and facility request ---------------------------------------------------
def stress(b, g=0.0, dm=0.0, dr=0.0, d_recv_days=0.0, new_debt=0.0, new_rate=0.0, tenor=5.0,
           revolving=False, tax_rate=0.25):
    """Simplified one-year pro-forma. b: dict with rev ebitda ebit da interest debt cur_debt cash recv inv pay cfo eq ni.
    g = revenue growth shock, dm = EBITDA-margin change (absolute, 0.02 = 2pp), dr = interest-rate change (0.01 = 1pp),
    d_recv_days = change in receivable days, new_debt/new_rate/tenor = proposed facility."""
    n = lambda k: 0.0 if (b.get(k) is None or pd.isna(b.get(k))) else float(b[k])
    rev, ebitda, ebit, da = n("rev"), n("ebitda"), n("ebit"), n("da")
    interest, debt, cur_debt, cash = n("interest"), n("debt"), n("cur_debt"), n("cash")
    recv, inv, pay, cfo, eq, ni = n("recv"), n("inv"), n("pay"), n("cfo"), n("eq"), n("ni")
    margin = ebitda / rev if rev else 0.0
    rev2 = rev * (1 + g)
    ebitda2 = rev2 * (margin + dm)
    ebit2 = ebitda2 - da
    r = interest / debt if debt > 0 else 0.0
    scale = max(r + dr, 0.0) / r if r > 0 else 1.0
    interest2 = interest * scale + new_debt * max(new_rate + dr, 0.0)
    rdays = recv / rev * 365 if rev else 0.0
    recv2 = (rdays + d_recv_days) * rev2 / 365
    k = rev2 / rev if rev else 1.0
    d_wc = (recv2 - recv) + (inv * k - inv) - (pay * k - pay)
    d_int = interest2 - interest
    d_tax = tax_rate * ((ebit2 - ebit) - d_int)
    cfo2 = cfo + (ebitda2 - ebitda) - d_int - d_wc - d_tax
    cfads2 = cfo2 + interest2
    ds2 = interest2 + cur_debt + (0.0 if revolving else new_debt / tenor if tenor else 0.0)
    ni2 = (ebit2 - interest2) * (1 - tax_rate)
    eq2 = eq + (ni2 - ni)
    debt2 = debt + new_debt
    res = dict(
        rev=rev2, ebitda=ebitda2, ebit=ebit2, interest=interest2, cfo=cfo2, cfads=cfads2, debt_service=ds2,
        icr=ebit2 / interest2 if interest2 > 0 else np.nan,
        dscr=cfads2 / ds2 if ds2 > 0 else np.nan,
        de=debt2 / eq2 if eq2 > 0 else np.nan,
        nd_ebitda=(debt2 - cash) / ebitda2 if ebitda2 > 0 else np.nan,
        cce=cfo2 / ebitda2 if ebitda2 > 0 else np.nan,
        runway_m=cash / (-cfo2 / 12) if cfo2 < 0 else np.nan,
        debt=debt2, equity=eq2)
    return res


def debt_capacity(b, target_dscr, rate, tenor, revolving=False):
    """Largest new facility (Rs cr) that keeps DSCR >= target. Conservative: ignores the tax shield on new interest."""
    n = lambda k: 0.0 if (b.get(k) is None or pd.isna(b.get(k))) else float(b[k])
    cfads = n("cfo") + n("interest")
    room = cfads / target_dscr - n("interest") - n("cur_debt")
    per = rate + (0.0 if revolving else 1.0 / tenor)
    return max(room / per, 0.0) if per > 0 else np.nan


def sens_grid(b, dm_list, dr_list, metric="dscr", **kw):
    """Grid of `metric` for EBITDA-margin changes (columns) and interest-rate changes (rows)."""
    return pd.DataFrame([[stress(b, dm=dm, dr=dr, **kw)[metric] for dm in dm_list] for dr in dr_list],
                        index=dr_list, columns=dm_list)


# ---- covenants --------------------------------------------------------------------------
COVENANT_SPECS = {   # metric key -> (label, kind, default)
    "icr": ("Interest cover (min)", "min", 2.0),
    "dscr": ("DSCR (min)", "min", 1.25),
    "de": ("Debt / Equity (max)", "max", 1.5),
    "nd_ebitda": ("Net debt / EBITDA (max)", "max", 3.5),
    "current_ratio": ("Current ratio (min)", "min", 1.2),
}


def covenant_status(actual, limit, kind, tight=0.15):
    """Return (status, headroom). Headroom is relative distance to the limit (negative = breached)."""
    if actual is None or pd.isna(actual) or not limit:
        return "n/a", np.nan
    head = (actual - limit) / abs(limit) if kind == "min" else (limit - actual) / abs(limit)
    return ("Breach" if head < 0 else "Tight" if head < tight else "OK"), head


# ---- cash flow: lifecycle and waterfall -------------------------------------------------
LIFECYCLE = {
    (False, False, True): ("Introduction", "Operations burn cash and are funded by new capital - high risk; check runway."),
    (True, False, True): ("Growth", "Operations fund part of the expansion but capital is still being raised - watch leverage."),
    (True, False, False): ("Mature", "Self-funding and repaying capital - usually the lowest credit risk."),
    (False, False, False): ("Shake-out", "Mixed signals: cash is draining from every activity - investigate."),
    (True, True, True): ("Shake-out", "Mixed signals: asset sales plus new funding - investigate."),
    (True, True, False): ("Shake-out", "Asset sales fund repayments - check whether it is sustainable."),
    (False, True, True): ("Decline", "Operations burn cash and assets are being sold - high risk."),
    (False, True, False): ("Decline", "Operations burn cash while capital is being returned - high risk."),
}


def lifecycle(cfo, cfi, cff):
    """Stage from the signs of operating / investing / financing cash flow. Returns (stage, reading) or (None, None)."""
    if any(v is None or pd.isna(v) for v in (cfo, cfi, cff)):
        return None, None
    return LIFECYCLE[(cfo > 0, cfi > 0, cff > 0)]


def waterfall_steps(row):
    """Where a year's cash came from and went. row: Series with cfo capex cfi cff divs netdebt. Returns (steps, net, approx)."""
    g = lambda k: (np.nan if k not in row.index else row[k])
    cfo, capex, cfi, cff, divs, nd = g("cfo"), g("capex"), g("cfi"), g("cff"), g("divs"), g("netdebt")
    z = lambda v: 0.0 if pd.isna(v) else float(v)
    approx = any(pd.isna(v) for v in (capex, divs, nd))
    other_inv = z(cfi) - z(capex) if not pd.isna(cfi) else 0.0
    other_fin = z(cff) - z(divs) - z(nd) if not pd.isna(cff) else 0.0
    steps = [("Operating cash flow", z(cfo)), ("Capex", z(capex)), ("Other investing", other_inv),
             ("Dividends", z(divs)), ("Net borrowing", z(nd)), ("Other financing", other_fin)]
    return steps, sum(v for _, v in steps), approx


# ---- return drivers and score breakdown -------------------------------------------------
def dupont(d):
    """DuPont and modified DuPont by year: ROE = NPM x asset turnover x equity multiplier;
    NPM = EBIT margin x interest burden x tax burden."""
    c = lambda k: d[k] if k in d else pd.Series(np.nan, index=d.index)
    ta_avg, eq_avg = avg(c("ta")), avg(c("equity"))
    out = pd.DataFrame(index=d.index)
    out["Net margin"] = div(c("ni"), c("rev"))
    out["Asset turnover"] = div(c("rev"), ta_avg)
    out["Equity multiplier"] = div(ta_avg, eq_avg)
    out["ROE"] = out["Net margin"] * out["Asset turnover"] * out["Equity multiplier"]
    out["EBIT margin"] = div(c("ebit"), c("rev"))
    out["Interest burden"] = div(c("pbt"), c("ebit"))
    out["Tax burden"] = div(c("ni"), c("pbt"))
    return out.replace([np.inf, -np.inf], np.nan)


def altman_components(d, year=None):
    """Weighted contribution of each term of Altman Z'' for one year: list of (label, contribution)."""
    year = year if year is not None else d.index[-1]
    r = d.loc[year]
    g = lambda k: r[k] if k in r.index else np.nan
    ta, tl = g("ta"), g("tl")
    x1, x2, x3, x4 = (g("ca") - g("cl")) / ta, g("retained") / ta, g("ebit") / ta, g("equity") / tl
    return [("Working capital / assets (x6.56)", 6.56 * x1), ("Retained earnings / assets (x3.26)", 3.26 * x2),
            ("EBIT / assets (x6.72)", 6.72 * x3), ("Equity / liabilities (x1.05)", 1.05 * x4)]
