"""Rule-based 'Insights & analysis' and credit-memo writer. Pure Python (no LLM): every sentence is derived
from the thresholds in metrics.py, so every sentence is reproducible and auditable."""
import numpy as np
import pandas as pd
import metrics as mt
from metrics import _q

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


def conditions(vals, stance_txt):
    """Conditions that match the stance: a 'Supportable' case gets actions only for concern-zone metrics and a light
    monitoring line for watch-zone ones; weaker stances also get the watch-zone actions."""
    reds = [k for k in mt.M if mt.rag(k, vals.get(k)) == "R"]
    ambers = [k for k in mt.M if mt.rag(k, vals.get(k)) == "A"]
    pool = reds if stance_txt == "Supportable" else reds + ambers
    acts = []
    for k in pool:
        a = ACTION.get(k)
        if a and a not in acts:
            acts.append(a)
    acts = acts[:6]
    if stance_txt == "Supportable" and ambers:
        acts.append("Monitor at the next review: " + ", ".join(mt.M[k]["label"] for k in ambers[:5]) + ".")
    return acts or ["Standard monitoring; refresh after the next results."]



# ====================================================================== ADVANCED: deeper analysis
TAB_OF = {"Liquidity": "Liquidity", "Solvency": "Solvency & debt service", "Returns": "Solvency & debt service", "Growth & margins": "Solvency & debt service",
          "Earnings quality": "Earnings quality", "Market": "Earnings quality", "Scores": "Distress & forensic"}
ZONE = {"G": 2, "A": 1, "R": 0}
ZONE_NAME = {"G": "healthy", "A": "watch", "R": "concern"}
SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Strength": 3}


def _cr(v):
    return "n/a" if v is None or pd.isna(v) else f"Rs {v:,.0f} cr"


def _chg(a, b):
    """Percentage change a -> b as text, or n/a."""
    if pd.isna(a) or pd.isna(b) or a == 0:
        return "n/a"
    return f"{(b - a) / abs(a):+.0%}"


def _g(fin, k, y):
    return fin.at[y, k] if (k in fin.columns and y in fin.index) else np.nan


def trends(m_df, peers_m=None):
    """Direction of every rated metric over the available history, biggest and zone-changing moves first.
    peers_m: {name: metric DataFrame} of sector companies, used to say how the move ranks in the sector."""
    out = []
    for k, meta in mt.M.items():
        if meta["dir"] is None or k not in m_df.index or meta["unit"] == "Rs cr":
            continue
        s = m_df.loc[k].dropna()
        if len(s) < 2:
            continue
        a, b = s.iloc[0], s.iloc[-1]
        if a == b:
            continue
        ra, rb = mt.rag(k, a), mt.rag(k, b)
        improving = (b > a) if meta["dir"] == "high" else (b < a)
        d = np.diff(s.values)
        steady = len(s) >= 3 and (all(d > 0) or all(d < 0))
        rel = min(abs(b - a) / (abs(a) + 1e-9), 5.0)
        rank_txt = None
        if peers_m:
            moves = []
            for nm, pm in peers_m.items():
                if k in pm.index:
                    ps = pm.loc[k].reindex(s.index).dropna()
                    if len(ps) >= 2 and ps.index[0] == s.index[0] and ps.index[-1] == s.index[-1]:
                        moves.append(ps.iloc[-1] - ps.iloc[0])
            if len(moves) >= 4:
                mv = b - a
                good = (lambda x: x) if meta["dir"] == "high" else (lambda x: -x)
                if good(mv) <= min(good(x) for x in moves) + 1e-12:
                    rank_txt = "the steepest deterioration in the sector"
                elif good(mv) >= max(good(x) for x in moves) - 1e-12:
                    rank_txt = "the biggest improvement in the sector"
        verb = "rose" if b > a else "fell"
        text = f"**{meta['label']}** {verb} from {fmt(k, a)} (FY{s.index[0]}) to {fmt(k, b)} (FY{s.index[-1]})"
        if ra and rb and ra != rb:
            text += f", moving from *{ZONE_NAME[ra]}* to *{ZONE_NAME[rb]}*"
        if steady:
            text += f" - {'up' if b > a else 'down'} every year"
        if rank_txt:
            text += f" - {rank_txt}"
        out.append(dict(key=k, a=a, b=b, ra=ra, rb=rb, first=s.index[0], last=s.index[-1], improving=improving,
                        zone=(ZONE[rb] - ZONE[ra]) if (ra and rb) else 0, rel=rel, steady=steady, sector=rank_txt, text=text + "."))
    out.sort(key=lambda r: (-(abs(r["zone"]) > 0), -(r["sector"] is not None), -r["rel"]))
    return out


def _sev(key, v):
    return {"R": "High", "A": "Medium"}.get(mt.rag(key, v), "Low")


def root_causes(fin, m_df):
    """Explain *why* key metrics moved, by tracing them back to the statement lines. List of dicts:
    title, severity, text, key (metric), evidence (list)."""
    if len(fin) < 2:
        return []
    y, p = fin.index[-1], fin.index[-2]
    g = lambda k, yy=y: _g(fin, k, yy)
    mv = lambda k, yy: m_df.at[k, yy] if (k in m_df.index and yy in m_df.columns) else np.nan
    out = []

    # 1. Cash conversion: which working-capital line absorbed the cash?
    cce_y, cce_p = mv("cce", y), mv("cce", p)
    if not pd.isna(cce_y) and (cce_y < mt.thresholds("cce")[0] - 0.03 or (not pd.isna(cce_p) and cce_y < cce_p - 0.10)):
        days = lambda num, den, yy: _q(g(num, yy), g(den, yy)) * 365
        drivers = []
        for lbl, num, den, sign in (("Receivable days", "recv", "rev", 1), ("Inventory days", "inv", "cogs", 1),
                                    ("Payable days", "pay", "cogs", -1)):
            dy, dp = days(num, den, y), days(num, den, p)
            if not (pd.isna(dy) or pd.isna(dp)):
                drivers.append((sign * (dy - dp), lbl, dp, dy, num))
        drivers.sort(reverse=True)
        ev = [f"CFO {_cr(g('cfo'))} vs EBITDA {_cr(g('ebitda'))} → {fmt('cce', cce_y)} converted"]
        if drivers and drivers[0][0] > 5:
            adv, lbl, dp, dy, num = drivers[0]
            line_g, rev_g = _chg(g(num, p), g(num)), _chg(g("rev", p), g("rev"))
            txt = (f"Cash conversion is {fmt('cce', cce_y)}" + (f" (was {fmt('cce', cce_p)})" if not pd.isna(cce_p) else "")
                   + f". The main drag is **{lbl.lower()}**, which went from {dp:,.0f} to {dy:,.0f} days"
                   + (f" ({'receivables' if num == 'recv' else 'inventory' if num == 'inv' else 'payables'} {line_g} vs sales {rev_g})." if num != "pay"
                      else " - suppliers are being paid faster, which uses cash."))
            ev += [f"{l}: {a:,.0f} → {b:,.0f} days" for _, l, a, b, _ in drivers]
        else:
            txt = (f"Cash conversion is {fmt('cce', cce_y)}, but receivable, inventory and payable days barely moved - "
                   "the gap is likely in non-working-capital items (tax paid, provisions, other income booked in EBITDA).")
        out.append(dict(title="Why profit is not turning into cash", key="cce", text=txt, evidence=ev,
                        severity=_sev("cce", cce_y)))

    # 2. Interest cover: profit problem or borrowing problem?
    icr_y, icr_p = mv("icr", y), mv("icr", p)
    if not (pd.isna(icr_y) or pd.isna(icr_p)) and icr_y < icr_p * 0.85 and icr_y < 2 * mt.thresholds("icr")[0]:
        e_ch = _q(g("ebit") - g("ebit", p), abs(g("ebit", p)))
        i_ch = _q(g("interest") - g("interest", p), abs(g("interest", p)))
        cause = ("mainly higher borrowing costs" if (not pd.isna(i_ch) and (pd.isna(e_ch) or i_ch > max(-e_ch, 0)))
                 else "mainly weaker operating profit")
        out.append(dict(title="Why interest cover fell", key="icr", severity=_sev("icr", icr_y),
                        text=f"Interest cover fell from {fmt('icr', icr_p)} to {fmt('icr', icr_y)}: EBIT {_chg(g('ebit', p), g('ebit'))} "
                             f"while interest cost {_chg(g('interest', p), g('interest'))} - **{cause}**.",
                        evidence=[f"EBIT {_cr(g('ebit', p))} → {_cr(g('ebit'))}", f"Interest {_cr(g('interest', p))} → {_cr(g('interest'))}"]))

    # 3. Leverage: more debt or less EBITDA?
    nd_y, nd_p = mv("nd_ebitda", y), mv("nd_ebitda", p)
    if not (pd.isna(nd_y) or pd.isna(nd_p)) and nd_y > nd_p + 0.5 and nd_y > 0:
        ndebt = lambda yy: g("debt", yy) - g("cash", yy)
        n_ch, e_ch = _q(ndebt(y) - ndebt(p), abs(ndebt(p))), _q(g("ebitda") - g("ebitda", p), abs(g("ebitda", p)))
        cause = "new borrowing / lower cash" if (not pd.isna(n_ch) and (pd.isna(e_ch) or n_ch > -e_ch)) else "falling EBITDA"
        out.append(dict(title="Why leverage went up", key="nd_ebitda", severity=_sev("nd_ebitda", nd_y),
                        text=f"Net debt / EBITDA rose from {fmt('nd_ebitda', nd_p)} to {fmt('nd_ebitda', nd_y)}, driven by **{cause}** "
                             f"(net debt {_chg(ndebt(p), ndebt(y))}, EBITDA {_chg(g('ebitda', p), g('ebitda'))}).",
                        evidence=[f"Net debt {_cr(ndebt(p))} → {_cr(ndebt(y))}", f"EBITDA {_cr(g('ebitda', p))} → {_cr(g('ebitda'))}"]))

    # 4. ROE: operations or leverage? (DuPont)
    dp = mt.dupont(fin)
    if len(dp) >= 2 and not dp[["ROE", "Net margin", "Asset turnover", "Equity multiplier"]].iloc[-2:].isna().any().any():
        r0, r1 = dp.iloc[-2], dp.iloc[-1]
        if abs(r1["ROE"] - r0["ROE"]) > 0.02:
            f_ = {k: np.log(r1[k] / r0[k]) for k in ("Net margin", "Asset turnover", "Equity multiplier") if r0[k] > 0 and r1[k] > 0}
            if f_:
                main = max(f_, key=lambda k: abs(f_[k]))
                up = r1["ROE"] > r0["ROE"]
                warn = up and main == "Equity multiplier"
                out.append(dict(title="What moved return on equity", key="roe", severity="Medium" if warn else "Low",
                                text=f"ROE {'rose' if up else 'fell'} from {r0['ROE']:.1%} to {r1['ROE']:.1%}, driven mainly by "
                                     f"**{main.lower()}** ({r0[main]:.2f} → {r1[main]:.2f})."
                                     + (" The improvement comes from more leverage, not better operations - a weaker kind of gain for a lender."
                                        if warn else ""),
                                evidence=[f"{k}: {r0[k]:.3f} → {r1[k]:.3f}" for k in ("Net margin", "Asset turnover", "Equity multiplier")]))

    # 5. Accruals: where did the non-cash profit go?
    gap = g("ni") - g("cfo")
    sl = mv("sloan", y)
    if not pd.isna(gap) and gap > 0 and (mt.rag("sloan", sl) in ("A", "R") or mt.rag("cfo_ni", mv("cfo_ni", y)) in ("A", "R")):
        d_recv, d_inv = g("recv") - g("recv", p), g("inv") - g("inv", p)
        wc_build = sum(v for v in (d_recv, d_inv) if not pd.isna(v) and v > 0)
        share = wc_build / gap if gap else np.nan
        out.append(dict(title="Where the non-cash profit went", key="sloan", severity=min(_sev("sloan", sl), _sev("cfo_ni", mv("cfo_ni", y)), key=SEV_ORDER.get),
                        text=f"Net profit exceeded operating cash by **{_cr(gap)}**. In the same year receivables rose {_cr(d_recv)} and "
                             f"inventory {_cr(d_inv)}"
                             + ((", which more than accounts for the gap." if share >= 1 else f", which together explain about {share:.0%} of the gap.") if not pd.isna(share) and share > 0
                                else "; working capital does not explain the gap, so check other accruals (provisions, deferred revenue, other income)."),
                        evidence=[f"Net income {_cr(g('ni'))}", f"CFO {_cr(g('cfo'))}", f"Δ receivables {_cr(d_recv)}", f"Δ inventory {_cr(d_inv)}"]))

    # 6. Liquidity squeeze: assets or liabilities?
    cr_y, cr_p = mv("current_ratio", y), mv("current_ratio", p)
    if not (pd.isna(cr_y) or pd.isna(cr_p)) and cr_y < cr_p - 0.15:
        cd_y, cd_p = g("cur_debt"), g("cur_debt", p)
        debt_note = (f" Short-term debt went from {_cr(cd_p)} to {_cr(cd_y)}, so part of the squeeze is debt falling due."
                     if not (pd.isna(cd_y) or pd.isna(cd_p)) and cd_y > cd_p * 1.25 else "")
        out.append(dict(title="Why short-term liquidity tightened", key="current_ratio",
                        severity=_sev("current_ratio", cr_y),
                        text=f"Current ratio fell from {fmt('current_ratio', cr_p)} to {fmt('current_ratio', cr_y)}: current assets "
                             f"{_chg(g('ca', p), g('ca'))} vs current liabilities {_chg(g('cl', p), g('cl'))}." + debt_note,
                        evidence=[f"Current assets {_cr(g('ca', p))} → {_cr(g('ca'))}", f"Current liabilities {_cr(g('cl', p))} → {_cr(g('cl'))}"]))

    # 7. Margin squeeze
    m_y, m_p = _q(g("ebitda"), g("rev")), _q(g("ebitda", p), g("rev", p))
    if not (pd.isna(m_y) or pd.isna(m_p)) and m_y < m_p - 0.02:
        c_y, c_p = _q(g("cogs"), g("rev")), _q(g("cogs", p), g("rev", p))
        s_y, s_p = _q(g("sga"), g("rev")), _q(g("sga", p), g("rev", p))
        parts = []
        if not (pd.isna(c_y) or pd.isna(c_p)):
            parts.append((c_y - c_p, f"cost of revenue went from {c_p:.1%} to {c_y:.1%} of sales"))
        if not (pd.isna(s_y) or pd.isna(s_p)):
            parts.append((s_y - s_p, f"SG&A went from {s_p:.1%} to {s_y:.1%} of sales"))
        parts.sort(reverse=True)
        out.append(dict(title="Why operating margin shrank", key=None, severity="Medium" if m_y < m_p - 0.05 else "Low",
                        text=f"EBITDA margin fell from {m_p:.1%} to {m_y:.1%}" + (f"; mainly because {parts[0][1]}." if parts and parts[0][0] > 0 else "."),
                        evidence=[t for _, t in parts]))
    return out


def peer_position(va, latest_all):
    """Rated metrics where the company sits in the sector's top or bottom quartile. Returns (top, bottom) lists of (key, pct)."""
    top, bottom = [], []
    for k, meta in mt.M.items():
        if meta["dir"] is None:
            continue
        pct = mt.percentile(k, va.get(k), [lv.get(k) for lv in latest_all.values()])
        if pd.isna(pct):
            continue
        if pct >= 75:
            top.append((k, pct))
        elif pct <= 25:
            bottom.append((k, pct))
    return sorted(top, key=lambda x: -x[1]), sorted(bottom, key=lambda x: x[1])


def fix_targets(fin):
    """For weak metrics: what change in the underlying numbers brings them to the healthy threshold.
    For healthy ones: how much cushion before they reach the concern threshold. Returns (fixes, cushions)."""
    if not len(fin):
        return [], []
    y = fin.index[-1]
    g = lambda k: _g(fin, k, y)
    debt, cash, ebitda, ebit, intr, cfo, cur, ca, cl, eq = (g(k) for k in
        ("debt", "cash", "ebitda", "ebit", "interest", "cfo", "cur_debt", "ca", "cl", "equity"))
    cur = 0.0 if pd.isna(cur) else cur
    fixes, cush = [], []

    nd = debt - cash
    if not (pd.isna(nd) or pd.isna(ebitda)) and ebitda > 0:
        v, tg, rd = nd / ebitda, *mt.thresholds("nd_ebitda")
        if v > tg:
            fixes.append(dict(key="nd_ebitda", now=v, target=tg,
                              text=f"Repay about **{_cr(nd - tg * ebitda)}** of net debt, or grow EBITDA by **{nd / tg / ebitda - 1:.0%}** "
                                   f"(to {_cr(nd / tg)})."))
        elif nd > 0:
            cush.append(dict(key="nd_ebitda", text=f"EBITDA can fall **{1 - nd / rd / ebitda:.0%}** before net debt / EBITDA reaches {rd:.0f}x."))
        else:
            cush.append(dict(key="nd_ebitda", text="Net cash position: cash exceeds total debt."))
    if not (pd.isna(ebit) or pd.isna(intr)) and intr > 0:
        v, tg, rd = ebit / intr, *mt.thresholds("icr")
        rate = _q(intr, debt)
        if v < tg:
            cut = 1 - ebit / tg / intr
            fixes.append(dict(key="icr", now=v, target=tg,
                              text=f"Raise EBIT by **{tg * intr / ebit - 1:.0%}** (to {_cr(tg * intr)}), or cut interest cost by **{cut:.0%}**"
                                   + (f" - about {_cr(cut * intr / rate)} of debt at the current {rate:.1%} rate." if not pd.isna(rate) and rate > 0 and ebit > 0 else ".")
                              if ebit > 0 else "EBIT is negative: interest is not covered at all; operating profit has to recover first."))
        else:
            cush.append(dict(key="icr", text=f"EBIT can fall **{1 - rd * intr / ebit:.0%}** before interest cover drops to {rd:.1f}x."))
    if not (pd.isna(cfo) or pd.isna(intr)):
        ds = intr + cur
        if ds > 0:
            v, tg, rd = (cfo + intr) / ds, *mt.thresholds("dscr")
            if v < tg:
                refi = (cfo + intr) / tg - intr
                fixes.append(dict(key="dscr", now=v, target=tg,
                                  text=f"Operating cash flow needs to reach **{_cr(tg * ds - intr)}** (now {_cr(cfo)})"
                                       + (f", or refinance short-term maturities down to about **{_cr(refi)}** (now {_cr(cur)})." if refi > 0
                                          else " - refinancing alone cannot fix this, because cash from operations does not even cover interest.")))
            elif cfo > 0:
                cush.append(dict(key="dscr", text=f"Operating cash flow can fall **{1 - (rd * ds - intr) / cfo:.0%}** before DSCR drops below {rd:.1f}x."))
    if not (pd.isna(ca) or pd.isna(cl)) and cl > 0:
        v, tg = ca / cl, mt.thresholds("current_ratio")[0]
        if v < tg:
            fixes.append(dict(key="current_ratio", now=v, target=tg,
                              text=f"Add **{_cr(tg * cl - ca)}** of current assets (e.g. fresh long-term funds held as cash), or cut current "
                                   f"liabilities by **{_cr(cl - ca / tg)}** (e.g. term out short-term borrowing)."))
    if not (pd.isna(debt) or pd.isna(eq)) and eq > 0:
        v, tg = debt / eq, mt.thresholds("de")[0]
        if v > tg:
            fixes.append(dict(key="de", now=v, target=tg,
                              text=f"Repay **{_cr(debt - tg * eq)}** of debt, or add **{_cr(debt / tg - eq)}** of equity."))
    if not (pd.isna(cfo) or pd.isna(ebitda)) and ebitda > 0:
        v, tg = cfo / ebitda, mt.thresholds("cce")[0]
        if v < tg and (tg * ebitda - cfo) > 0.02 * ebitda:
            fixes.append(dict(key="cce", now=v, target=tg,
                              text=f"Release about **{_cr(tg * ebitda - cfo)}** of cash from working capital (faster collections, leaner stock) "
                                   f"to bring CFO to {tg:.0%} of EBITDA."))
    return fixes, cush


def cards(name, va, m_df, fin, flags, trend_rows, causes, vb=None, bench_label=None, latest_all=None):
    """Prioritised insight cards. Each: severity, title, body, evidence (list), why, ask, tab."""
    out, covered = [], set()
    rated = {k: mt.rag(k, va.get(k)) for k in mt.M}

    def ctx(k):
        ev = []
        if vb is not None and not pd.isna(vb.get(k, np.nan)):
            ev.append(f"{bench_label}: {fmt(k, vb.get(k))}")
        if latest_all:
            pct = mt.percentile(k, va.get(k), [lv.get(k) for lv in latest_all.values()])
            if not pd.isna(pct):
                ev.append(f"Sector percentile: {pct:.0f}")
        return ev

    for c in causes:
        k = c["key"]
        out.append(dict(severity=c["severity"], title=c["title"], body=c["text"], evidence=c["evidence"] + (ctx(k) if k else []),
                        why=WHY.get(k, ""), ask=ACTION.get(k, ""), tab=TAB_OF[mt.M[k]["group"]] if k else "Solvency & debt service"))
        if k:
            covered.add(k)
    for t in trend_rows:
        if t["zone"] < 0 and t["key"] not in covered:
            k = t["key"]
            out.append(dict(severity="High" if t["rb"] == "R" else "Medium", title=f"{mt.M[k]['label']} is deteriorating",
                            body=t["text"], evidence=ctx(k), why=WHY.get(k, ""), ask=ACTION.get(k, ""), tab=TAB_OF[mt.M[k]["group"]]))
            covered.add(k)
    for sev, text in flags:
        out.append(dict(severity="High" if sev == "R" else "Medium", title="Red flag: " + text.split(" - ")[0].rstrip("."),
                        body=text, evidence=[],
                        why="", ask="", tab="Distress & forensic"))
    for k in REASON_ORDER + [k for k in mt.M if k not in REASON_ORDER]:
        if rated.get(k) == "R" and k not in covered:
            out.append(dict(severity="High" if k in KEY_STRENGTH or k in DISTRESS else "Medium",
                            title=f"{mt.M[k]['label']} in the concern zone", body=f"{mt.M[k]['label']} is **{fmt(k, va[k])}**. {WHY.get(k, '')}",
                            evidence=[f"Formula: {mt.M[k]['formula']}"] + ctx(k), why="", ask=ACTION.get(k, ""), tab=TAB_OF[mt.M[k]["group"]]))
            covered.add(k)
    for t in trend_rows:
        if t["zone"] > 0 or (t["improving"] and t["sector"]):
            k = t["key"]
            out.append(dict(severity="Strength", title=f"{mt.M[k]['label']} is improving", body=t["text"], evidence=ctx(k),
                            why=WHY.get(k, ""), ask="", tab=TAB_OF[mt.M[k]["group"]]))
            covered.add(k)
    if latest_all:
        top, _ = peer_position(va, latest_all)
        for k, pct in top:
            if k in KEY_STRENGTH and rated.get(k) == "G" and k not in covered:
                out.append(dict(severity="Strength", title=f"Sector leader: {mt.M[k]['label']}",
                                body=f"{mt.M[k]['label']} of **{fmt(k, va[k])}** beats {pct:.0f}% of the sector.", evidence=ctx(k),
                                why=WHY.get(k, ""), ask="", tab=TAB_OF[mt.M[k]["group"]]))
                covered.add(k)
    out.sort(key=lambda c: SEV_ORDER[c["severity"]])
    return out
