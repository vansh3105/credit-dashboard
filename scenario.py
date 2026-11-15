"""What-if on the latest year: revenue, EBITDA margin and interest-rate shocks, and the points where cover breaks.

Rules for the shocked year (everything else stays as reported):
  revenue x (1 + g); EBITDA = new revenue x (margin + dm); EBIT = EBITDA - depreciation
  interest = interest + debt x dr (the whole debt book reprices)
  other income below EBIT and the tax rate stay as reported; net profit follows
  operating cash flow moves by the change in EBITDA after tax
  receivables, inventory and payables move with revenue (days unchanged); current assets/liabilities follow
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics as mt


def _g(row, k):
    v = row.get(k, np.nan)
    return np.nan if v is None else float(v)


def shock_row(row: pd.Series, g=0.0, dm=0.0, dr=0.0) -> pd.Series:
    r = row.copy()
    rev, ebitda, da = _g(row, "rev"), _g(row, "ebitda"), _g(row, "da")
    intr, debt, pbt, tax, ebit = _g(row, "interest"), _g(row, "debt"), _g(row, "pbt"), _g(row, "tax"), _g(row, "ebit")
    if any(np.isnan(x) for x in (rev, ebitda)) or rev == 0:
        return r
    da = 0.0 if np.isnan(da) else da
    margin = ebitda / rev
    rev2 = rev * (1 + g)
    ebitda2 = rev2 * (margin + dm)
    ebit2 = ebitda2 - da
    intr0 = 0.0 if np.isnan(intr) else intr
    intr2 = intr0 + (0.0 if np.isnan(debt) else debt) * dr
    other = (pbt - (ebit - intr0)) if not any(np.isnan(x) for x in (pbt, ebit)) else 0.0
    pbt2 = ebit2 - intr2 + other
    t = float(np.clip(tax / pbt, 0, 0.4)) if not any(np.isnan(x) for x in (tax, pbt)) and pbt > 0 else 0.25
    r["rev"], r["ebitda"], r["ebit"], r["interest"], r["pbt"] = rev2, ebitda2, ebit2, intr2, pbt2
    r["tax"] = max(pbt2, 0) * t
    r["ni"] = pbt2 - r["tax"]
    if not np.isnan(_g(row, "cfo")):
        r["cfo"] = _g(row, "cfo") + (ebitda2 - ebitda) * (1 - t) - (intr2 - intr0) * (1 - t)
    if not np.isnan(_g(row, "cogs")) and rev - ebitda:
        r["cogs"] = _g(row, "cogs") * (rev2 - ebitda2) / (rev - ebitda)
    d_ca = d_cl = 0.0
    for k in ("recv", "inv"):
        if not np.isnan(_g(row, k)):
            r[k] = _g(row, k) * (1 + g)
            d_ca += r[k] - _g(row, k)
    if not np.isnan(_g(row, "pay")):
        r["pay"] = _g(row, "pay") * (1 + g)
        d_cl += r["pay"] - _g(row, "pay")
    if not np.isnan(_g(row, "ca")):
        r["ca"] = _g(row, "ca") + d_ca
    if not np.isnan(_g(row, "cl")):
        r["cl"] = _g(row, "cl") + d_cl
    return r


def run(fin: pd.DataFrame, mcap_cr=np.nan, gnp=100.0, gov=None, weights=None, g=0.0, dm=0.0, dr=0.0) -> dict:
    """Metrics, score and grade before and after the shock. Returns dict(base, shocked, score0, score1, grade0, grade1)."""
    import insights as ins
    import report as rp
    f2 = fin.copy()
    f2.iloc[-1] = shock_row(fin.iloc[-1], g, dm, dr)
    v0 = ins.latest(mt.compute(fin, mcap_cr, gnp))
    v1 = ins.latest(mt.compute(f2, mcap_cr, gnp))
    s0 = mt.overall_score(mt.pillar_scores(v0, gov), weights)
    s1 = mt.overall_score(mt.pillar_scores(v1, gov), weights)
    return dict(base=v0, shocked=v1, score0=s0, score1=s1, grade0=rp.grade_index(s0), grade1=rp.grade_index(s1))


def _cover(row, g, dm, dr):
    r = shock_row(row, g, dm, dr)
    intr, cur = _g(r, "interest"), _g(r, "cur_debt")
    cur = 0.0 if np.isnan(cur) else cur
    dscr = (_g(r, "cfo") + intr) / (intr + cur) if (intr + cur) > 0 else np.nan
    icr = _g(r, "ebit") / intr if intr > 0 else np.nan
    return dscr, icr


def breakpoints(fin: pd.DataFrame, dm=0.0, dr=0.0) -> dict:
    """Smallest revenue fall (0-90%, 1% steps) at which DSCR drops below 1.25 and 1.0 and interest cover below 1.5,
    with the margin and rate shocks applied. None = does not break within a 90% fall; 0 = already below today."""
    row = fin.iloc[-1]
    out = {}
    for name, idx, cut in (("DSCR < 1.25x", 0, 1.25), ("DSCR < 1.0x", 0, 1.0), ("Interest cover < 1.5x", 1, 1.5)):
        hit = None
        for pct in range(0, 91):
            v = _cover(row, -pct / 100, dm, dr)[idx]
            if not np.isnan(v) and v < cut:
                hit = pct
                break
        out[name] = hit
    return out
