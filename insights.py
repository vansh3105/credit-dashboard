"""Rule-based 'Insights & analysis' and credit-memo writer. Pure Python (no LLM): every sentence is derived
from the thresholds in metrics.py, so every sentence is reproducible and auditable."""
import numpy as np
import pandas as pd
import metrics as mt

WHY = {
    "current_ratio": "A comfortable current ratio can hide slow-moving stock or stuck receivables.",
    "quick_ratio": "Strips out inventory, so it tests whether near-cash assets cover short-term dues.",
    "wc_days": "Days of sales blocked in working capital - why a profitable firm can still need bigger limits.",
    "ocf_ratio": "Is liquidity coming from real operations? Cash pays debt, profit does not.",
    "cce": "How much of every Rs 100 of EBITDA turns into operating cash.",
    "dir_days": "Days the firm can pay its cash costs with zero new revenue.",
    "runway_m": "Burning cash: months until cash runs out.",
    "de": "Leverage: how much of the balance sheet is lender-funded.",
    "debt_ratio": "Share of assets financed by debt.",
    "icr": "EBIT cushion over interest; around 1.5x the firm looks highly leveraged.",
    "dscr": "Below 1 means cash flow cannot cover interest plus principal due.",
    "nd_ebitda": "Years of EBITDA needed to repay net debt; above about 4x is stretched.",
    "sloan": "Share of profit that is accruals, not cash - high values mean low earnings quality.",
    "cf_accrual": "Why didn't profits become cash?",
    "bs_accrual": "Where did the profits go? Large growth in net operating assets means receivables/inventory.",
    "cfo_ni": "Operating cash should at least match profit.",
    "ffo_tl": "Operating funds generated per rupee of liabilities.",
    "roic_spread": "Returns vs cost of debt: below zero means growth does not cover what the lenders are paid.",
    "altman_z": "Altman Z (needs market cap): below 1.81 is distress, above 2.99 safe.",
    "altman_zp": "Altman Z' for private manufacturers.",
    "altman_z2": "Altman Z'' for non-manufacturers / emerging markets.",
    "ohlson_p": "Ohlson estimated probability of distress; above 50% means default within two years.",
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
    "nd_ebitda": "Cap further borrowing and consider a net-debt/EBITDA covenant.",
    "de": "Verify the source and quality of debt and equity (loan-to-equity conversions can flatter ratios).",
    "altman_z2": "Tighten covenants / collateral cover; treat as watch-list candidate.",
    "altman_zp": "Tighten covenants / collateral cover; treat as watch-list candidate.",
    "altman_z": "Tighten covenants / collateral cover; treat as watch-list candidate.",
    "ohlson_p": "Treat as high-default-risk until cash generation is verified.",
    "beneish": "Run forensic checks: related parties, bank trail, GST/customer verification.",
    "roic_spread": "Question the return on incremental capex versus the interest rate being paid.",
}
KEY_STRENGTH = ["current_ratio", "cce", "icr", "dscr", "de", "nd_ebitda", "ocf_ratio", "altman_z2", "ohlson_p", "piotroski",
                "roic_spread", "sloan"]
DISTRESS = ["altman_z", "altman_zp", "altman_z2", "ohlson_p", "zmijewski", "springate", "beneish"]
REASON_ORDER = ["cce", "ocf_ratio", "dscr", "icr", "nd_ebitda", "de", "ohlson_p", "altman_z2", "altman_zp", "altman_z",
                "sloan", "cf_accrual", "beneish", "current_ratio", "dir_days", "wc_days", "springate", "zmijewski"]


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


def verdict(vals, flags):
    """Headline signal for a company: dict(signal, g, a, r, reasons)."""
    rated = {k: mt.rag(k, vals.get(k)) for k in mt.M}
    reds = [k for k, r in rated.items() if r == "R"]
    g = sum(1 for r in rated.values() if r == "G")
    a = sum(1 for r in rated.values() if r == "A")
    dist_red = [k for k in DISTRESS if rated.get(k) == "R"]
    cfo_neg = any("negative" in t for s, t in flags)
    if len(reds) >= 6 or len(dist_red) >= 3 or cfo_neg:
        signal = "ELEVATED CONCERN"
    elif len(reds) >= 3 or dist_red:
        signal = "WATCH"
    else:
        signal = "COMFORTABLE"
    ordered = [k for k in REASON_ORDER if k in reds] + [k for k in reds if k not in REASON_ORDER]
    reasons = [t.split(" - ")[0] for s, t in flags if s == "R"][:1]
    reasons += [f"{mt.M[k]['label']}: {fmt(k, vals[k])}" for k in ordered]
    return dict(signal=signal, g=g, a=a, r=len(reds), reasons=reasons[:3])


def build(name_a, va, flags, bench_label=None, vb=None, name_b=None, benford_a=None):
    """va, vb: latest-value Series. Returns list of (heading, [bullet strings])."""
    v = verdict(va, flags)
    rated = {k: mt.rag(k, va.get(k)) for k in mt.M}
    reds = [k for k, r in rated.items() if r == "R"]
    ambers = [k for k, r in rated.items() if r == "A"]
    out = [("Overall signal", [
        f"**{name_a}: {v['signal']}** - {v['g']} green, {v['a']} amber, {v['r']} red across the rated metrics.",
        "This is a rule-based signal built from the metric thresholds, not a credit rating or a lending decision; "
        "the analyst still has to judge business model, management and collateral."])]
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
            rows.append((abs(va[k] - vb[k]) / (abs(vb[k]) + 1e-9), k, bt))
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
    out.append(("What to ask / consider next", acts[:6] or ["Standard monitoring; re-run after the next results."]))
    return out


# ====================================================================== ADVANCED: credit memo
def stance(signal, pro_dscr=None, breaches=0, target=1.25):
    """Indicative stance used in the memo."""
    if signal == "ELEVATED CONCERN" or (pro_dscr is not None and not pd.isna(pro_dscr) and pro_dscr < 1.0):
        return "Not supportable on current numbers", "Cash flow does not comfortably cover obligations; resolve the red items first."
    if signal == "WATCH" or breaches or (pro_dscr is not None and not pd.isna(pro_dscr) and pro_dscr < target):
        return "Supportable with conditions", "Proceed only with tighter monitoring, covenants and verification of the flagged items."
    return "Supportable", "Metrics are comfortable; keep standard monitoring."


def memo(name, sector, as_of, vals, flags, pillars, score, band_txt, snap_rows, req=None, scen=None, cov_rows=None,
         bf=None, cmp_name=None):
    """Return the credit-appraisal note as markdown. All inputs are pre-computed by the app."""
    v = verdict(vals, flags)
    lines = [f"# Credit appraisal note - {name}", "",
             f"*{sector} · {as_of} · draft for analyst review - generated from public data, not a credit decision*", ""]
    lines += ["## 1. Borrower and request", f"- Borrower: **{name}** ({sector})"]
    if req:
        lines += [f"- Request: **{req['type']}** of **Rs {req['amount']:,.0f} cr** at **{req['rate']:.1%}**"
                  + (f" over {req['tenor']:.0f} years" if req["type"].startswith("Term") else " (revolving)"),
                  f"- Largest facility keeping DSCR at or above {req['target']:.2f}x: **Rs {req['capacity']:,.0f} cr**"]
    else:
        lines += ["- Request: none entered (monitoring view)"]
    lines += ["", "## 2. Financial snapshot", "", "| Metric | " + " | ".join(snap_rows["cols"]) + " |",
              "|---|" + "---|" * len(snap_rows["cols"])]
    for label, vals_ in snap_rows["rows"]:
        lines.append(f"| {label} | " + " | ".join(vals_) + " |")
    lines += ["", "## 3. Indicative risk score",
              f"- Overall: **{score:.0f}/100 - {band_txt}** (rule-based; adjustable weights)",
              "- Pillars: " + "; ".join(f"{p} {s:.0f}" for p, s in pillars.items() if not pd.isna(s)),
              f"- Headline signal: **{v['signal']}** ({v['g']} green / {v['a']} amber / {v['r']} red)", ""]
    strengths = [f"{mt.M[k]['label']} {fmt(k, vals[k])}" for k in KEY_STRENGTH if mt.rag(k, vals.get(k)) == "G"][:5]
    concerns = [f"{mt.M[k]['label']} {fmt(k, vals[k])}" for k in mt.M if mt.rag(k, vals.get(k)) == "R"][:8]
    lines += ["## 4. Strengths", *([f"- {s}" for s in strengths] or ["- None clearly in the green zone"]), "",
              "## 5. Risks and red flags", *([f"- {c}" for c in concerns] or ["- No red metrics"]),
              *[f"- {t}" for s, t in flags]]
    if bf and bf.get("flag") is not None:
        lines.append(f"- Benford screen: KS {bf['ks']:.3f} vs cut-off {bf['cutoff']:.3f} - "
                     + ("unusual leading-digit pattern, verify source data" if bf["flag"] else "no unusual pattern"))
    lines.append("")
    if scen:
        lines += ["## 6. Pro-forma and stress", "", "| | Current | With facility | Stressed |", "|---|---|---|---|"]
        for label, k, f in (("Interest cover", "icr", "{:.2f}x"), ("DSCR", "dscr", "{:.2f}x"), ("Debt / Equity", "de", "{:.2f}x"),
                            ("Net debt / EBITDA", "nd_ebitda", "{:.2f}x"), ("Operating cash flow (Rs cr)", "cfo", "{:,.0f}")):
            cells = [("n/a" if pd.isna(scen[c][k]) else f.format(scen[c][k])) for c in ("base", "pro", "stress")]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", f"*Stress case: {scen['desc']}*", ""]
    if cov_rows:
        lines += ["## 7. Covenant headroom (latest year)", "", "| Covenant | Actual | Limit | Status |", "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in cov_rows]
        lines.append("")
    breaches = sum(1 for r in (cov_rows or []) if r[3] == "Breach")
    st_, why = stance(v["signal"], scen["pro"]["dscr"] if scen else None, breaches, req["target"] if req else 1.25)
    acts = []
    for k in [k for k in mt.M if mt.rag(k, vals.get(k)) in ("R", "A")]:
        a = ACTION.get(k)
        if a and a not in acts:
            acts.append(a)
    lines += ["## 8. Indicative stance", f"**{st_}.** {why}", "", "### Conditions / next steps",
              *([f"- {a}" for a in acts[:6]] or ["- Standard monitoring; refresh after next results."]), "",
              "---", "*Limitations: annual public statements only; DSCR is a proxy; no collateral, management or "
              "industry assessment is included. Figures in Rs crore.*"]
    return "\n".join(lines)
