"""Rule-based 'Insights & analysis' writer. Pure Python (no LLM): every sentence is derived from the
thresholds in metrics.py, so every sentence is reproducible and auditable."""
import numpy as np
import pandas as pd
import metrics as mt

WHY = {
    "current_ratio": "A comfortable current ratio can hide slow-moving stock or stuck receivables.",
    "quick_ratio": "Strips out inventory, so it tests whether near-cash assets cover short-term dues.",
    "wc_days": "Days of sales blocked in working capital - the reason a profitable firm still asks for bigger limits.",
    "ocf_ratio": "Is liquidity coming from real operations? Cash pays debt, profit does not.",
    "cce": "How much of every Rs 100 of EBITDA turns into operating cash.",
    "dir_days": "Days the firm can pay its cash costs with zero new revenue.",
    "runway_m": "Burning cash: months until cash runs out.",
    "de": "Leverage: how much of the balance sheet is lender-funded.",
    "debt_ratio": "Share of assets financed by debt.",
    "icr": "EBIT cushion over interest; around 1.5x the firm looks highly leveraged.",
    "dscr": "Below 1 means cash flow cannot cover interest plus principal due.",
    "sloan": "Share of profit that is accruals, not cash - high values mean low earnings quality.",
    "cf_accrual": "Why didn't profits become cash?.",
    "bs_accrual": "Where did the profits go? Large growth in net operating assets means receivables/inventory.",
    "cfo_ni": "Piotroski test: operating cash should at least match profit.",
    "ffo_tl": "Operating funds generated per rupee of liabilities.",
    "roic_spread": "Returns vs cost of debt: below zero means growth destroys value for lenders too.",
    "altman_z": "Altman Z (needs market cap): <1.81 distress, >2.99 safe.",
    "altman_zp": "Altman Z' for private manufacturers.",
    "altman_z2": "Altman Z'' for non-manufacturers / emerging markets.",
    "ohlson_p": "Ohlson estimated probability of distress; >50% = default within two years.",
    "zmijewski": "Zmijewski score: higher means higher bankruptcy risk.",
    "springate": "Springate: below 0.862 signals financial stress.",
    "piotroski": "Piotroski F-score: 7+ strong, 3 or less weak.",
    "beneish": "Beneish M-score above -2.22 suggests likely earnings manipulation.",
}
ACTION = {
    "cce": "Ask for a CFO-vs-EBITDA bridge and customer-wise receivable ageing before sanctioning working-capital limits.",
    "ocf_ratio": "Request monthly cash-flow forecasts; consider tighter drawing-power / stock-statement monitoring.",
    "sloan": "Investigate receivables and inventory build-up; check provisioning for doubtful debts.",
    "cf_accrual": "Investigate receivables and inventory build-up; check provisioning for doubtful debts.",
    "wc_days": "Test whether the extra limit funds genuine growth or just stuck receivables/stock.",
    "icr": "Stress-test EBIT falls of 20-30% against interest; consider covenants on coverage.",
    "dscr": "Reschedule or sculpt repayments to match the cash-flow pattern; set a minimum-DSCR covenant.",
    "de": "Verify the source and quality of debt and equity (loan-to-equity conversions can flatter ratios).",
    "altman_z2": "Tighten covenants / collateral cover; treat as watch-list candidate.",
    "altman_zp": "Tighten covenants / collateral cover; treat as watch-list candidate.",
    "altman_z": "Tighten covenants / collateral cover; treat as watch-list candidate.",
    "ohlson_p": "Treat as high-default-risk until cash generation is verified.",
    "beneish": "Run forensic checks: related parties, bank trail, GST/customer verification.",
    "roic_spread": "Question the return on incremental capex versus the interest rate being paid.",
}
KEY_STRENGTH = ["current_ratio", "cce", "icr", "dscr", "de", "ocf_ratio", "altman_z2", "ohlson_p", "piotroski", "roic_spread", "sloan"]
DISTRESS = ["altman_z", "altman_zp", "altman_z2", "ohlson_p", "zmijewski", "springate", "beneish"]


def latest(m: pd.DataFrame) -> pd.Series:
    """Last non-NaN value of each metric row."""
    return m.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)


def fmt(key, v):
    if v is None or pd.isna(v):
        return "n/a"
    u = mt.M[key]["unit"]
    return {"%": f"{v:.1%}", "x": f"{v:.2f}x", "days": f"{v:,.0f} d", "months": f"{v:,.0f} m",
            "pts": f"{v:.0f}/9", "Rs cr": f"Rs {v:,.0f} cr"}.get(u, f"{v:,.2f}")


def better(key, a, b):
    d = mt.M[key]["dir"]
    if d is None or pd.isna(a) or pd.isna(b) or a == b:
        return None
    return (a > b) if d == "high" else (a < b)


def build(name_a, va, flags, bench_label=None, vb=None, name_b=None, benford_a=None):
    """va, vb: latest-value Series. Returns list of (heading, [bullet strings])."""
    rated = {k: mt.rag(k, va.get(k)) for k in mt.M}
    reds = [k for k, r in rated.items() if r == "R"]
    ambers = [k for k, r in rated.items() if r == "A"]
    greens = [k for k, r in rated.items() if r == "G"]
    dist_red = [k for k in DISTRESS if rated.get(k) == "R"]
    cfo_neg = any("negative" in t for s, t in flags)
    if len(reds) >= 6 or len(dist_red) >= 3 or cfo_neg:
        signal = "ELEVATED CONCERN"
    elif len(reds) >= 3 or dist_red:
        signal = "WATCH"
    else:
        signal = "COMFORTABLE"
    out = []
    out.append(("Overall signal", [
        f"**{name_a}: {signal}** - {len(greens)} green, {len(ambers)} amber, {len(reds)} red across the rated metrics.",
        "This is a rule-based signal built from the metric thresholds, not a credit rating or a lending decision; "
        "the analyst still has to judge business model, management and collateral."]))
    s = [f"{mt.M[k]['label']}: {fmt(k, va[k])} - {WHY.get(k, '')}" for k in KEY_STRENGTH if rated.get(k) == "G"][:5]
    out.append(("What looks sound", s or ["No key metric is clearly in the green zone."]))
    c = [f"{mt.M[k]['label']}: {fmt(k, va[k])} - {WHY.get(k, '')}" for k in reds[:8]]
    c += [f"[{'High' if sv == 'R' else 'Medium'}] {t}" for sv, t in flags]
    out.append(("What worries a lender", c or ["No red metrics or flags triggered on the available data."]))
    if benford_a and benford_a.get("flag") is not None:
        out.append(("Benford's-law screen", [
            f"KS = {benford_a['ks']:.3f} vs cut-off {benford_a['cutoff']:.3f} on {benford_a['n']} reported figures: "
            + ("**above cut-off - the digit pattern is unusual; use it to decide where to dig, it is not proof of fraud**."
               if benford_a["flag"] else "within the cut-off - no unusual leading-digit pattern.")]))
    if vb is not None:
        rows = []
        for k in mt.M:
            if mt.M[k]["dir"] is None or pd.isna(va.get(k)) or pd.isna(vb.get(k)):
                continue
            bt = better(k, va[k], vb[k])
            if bt is None:
                continue
            rel = abs(va[k] - vb[k]) / (abs(vb[k]) + 1e-9)
            rows.append((rel, k, bt))
        rows.sort(reverse=True)
        worse = [f"{mt.M[k]['label']}: {fmt(k, va[k])} vs {fmt(k, vb[k])}" for _, k, bt in rows if not bt][:4]
        bet = [f"{mt.M[k]['label']}: {fmt(k, va[k])} vs {fmt(k, vb[k])}" for _, k, bt in rows if bt][:4]
        out.append((f"Versus {bench_label}", [
            "**Better than comparator:** " + ("; ".join(bet) or "none material"),
            "**Weaker than comparator:** " + ("; ".join(worse) or "none material")]))
    acts = []
    for k in reds + ambers:
        a = ACTION.get(k)
        if a and a not in acts:
            acts.append(a)
    out.append(("What to ask / consider next", acts[:6] or ["Standard monitoring; re-run after next quarterly results."]))
    return out
