import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as dt
import insights as ins
import metrics as mt
from sectors import SECTORS

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Credit Risk Analyst Dashboard", layout="wide", page_icon="📊")

C1, C2, CB = "#1f6feb", "#e8710a", "#8b949e"          # company 1, company 2, benchmark
RAG_TXT = {"G": "🟢 Green", "A": "🟡 Amber", "R": "🔴 Red", None: "-"}
GROUP_TITLES = {
    "Liquidity": "Can the borrower pay short-term dues from real cash?",
    "Solvency": "Can it carry its debt?",
    "Earnings quality": "Is profit backed by cash?",
    "Returns": "Are returns above the cost of capital?",
    "Scores": "Distress and manipulation scores",
    "Market": "What does the market price imply?",
}

st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1400px}
div[data-testid="stMetric"]{background:rgba(128,128,128,.10);padding:10px 14px;border-radius:10px}
.coname{font-weight:700;font-size:1.05rem;margin:.2rem 0 .3rem 0}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_sector(sector: str):
    return dt.load_sector(SECTORS[sector])


def bench_frame(peer_metrics: dict, how: str):
    frames = [m.reindex(list(mt.M)) for m in peer_metrics.values()]
    years = sorted(set().union(*[f.columns for f in frames]))
    arr = np.stack([f.reindex(columns=years).values.astype(float) for f in frames])
    fn = np.nanmedian if how == "Median" else np.nanmean
    with np.errstate(all="ignore"):
        res = fn(arr, axis=0)
    return pd.DataFrame(res, index=list(mt.M), columns=years)


fmtv = ins.fmt


def counts(vals, keys):
    r = [mt.rag(k, vals.get(k)) for k in keys]
    return r.count("G"), r.count("A"), r.count("R")


# ------------------------------------------------------------------ sidebar: the 3-step flow
st.sidebar.title("📊 Credit Risk Dashboard")
st.sidebar.caption("For the **Credit Risk Analyst** · live data from Yahoo Finance")
sector = st.sidebar.selectbox("1 · Sector", list(SECTORS))
mode = st.sidebar.selectbox("2 · Analysis type", ["Individual stats vs Industry average", "Peer company review"])

with st.spinner(f"Loading {sector} companies from Yahoo Finance (first load can take a minute)…"):
    companies, errors = get_sector(sector)

names = {t: c.name for t, c in companies.items()}
if not names:
    st.error("No company data could be loaded for this sector.")
    if errors:
        st.write({SECTORS[sector].get(t, t): e for t, e in errors.items()})
    st.info("Yahoo Finance may be unreachable or rate-limiting. Click **Reload data** in the sidebar and retry in a few minutes.")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if mode.startswith("Individual"):
    t1 = st.sidebar.selectbox("3 · Company", list(names), format_func=lambda t: f"{names[t]} ({t})")
    t2 = None
else:
    t1 = st.sidebar.selectbox("3 · Choose company 1", list(names), format_func=lambda t: f"{names[t]} ({t})")
    rest = [t for t in names if t != t1]
    if not rest:
        st.error("Need at least two loaded companies for a peer review.")
        st.stop()
    t2 = st.sidebar.selectbox("   Choose company 2", rest, format_func=lambda t: f"{names[t]} ({t})")

with st.sidebar.expander("Settings"):
    how = st.radio("Industry average =", ["Median", "Mean"], horizontal=True)
    gnp = st.number_input("Price index for size term (Ohlson)", 1.0, 1000.0, 100.0,
                          help="Total assets (Rs crore) are divided by this index in the Ohlson size term.")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()
if errors:
    st.sidebar.warning(f"{len(errors)} ticker(s) failed to load: " + ", ".join(SECTORS[sector].get(t, t) for t in errors))

# ------------------------------------------------------------------ compute
all_m = {t: mt.compute(c.fin, c.mcap_cr, gnp) for t, c in companies.items()}
A = companies[t1]; mA = all_m[t1]
yearA = mA.columns[-1]
vA = ins.latest(mA)
flagsA = mt.red_flags(A.fin, mA)
if t2 is None:
    peers = {t: m for t, m in all_m.items() if t != t1}
    if not peers:
        st.error("Industry average needs at least one other company in this sector to load.")
        st.stop()
    cmp_df = bench_frame(peers, how)
    cmp_name = f"Industry {how.lower()}"
    cmp_label = f"industry {how.lower()} (excl. {A.name}, {len(peers)} peers)"
    B = None
    vB = cmp_df[yearA] if yearA in cmp_df.columns else ins.latest(cmp_df)
    flagsB = []
else:
    B = companies[t2]; cmp_df = all_m[t2]; cmp_name = B.name; cmp_label = B.name
    vB = ins.latest(cmp_df)
    flagsB = mt.red_flags(B.fin, cmp_df)

series = {A.name: mA, cmp_name: cmp_df.reindex(list(mt.M))}
rawA = [x for y in A.raw_values for x in A.raw_values[y]]
bfA = mt.benford(rawA) if rawA else None
bfB = None
if B is not None:
    rawB = [x for y in B.raw_values for x in B.raw_values[y]]
    bfB = mt.benford(rawB) if rawB else None


# ------------------------------------------------------------------ tables
def latest_table(groups):
    rows = []
    for k, m in mt.M.items():
        if m["group"] not in groups:
            continue
        a, b = vA.get(k), vB.get(k)
        row = {"Metric": m["label"], A.name: fmtv(k, a), f"{A.name} signal": RAG_TXT[mt.rag(k, a)],
               cmp_name: fmtv(k, b), f"{cmp_name} signal": RAG_TXT[mt.rag(k, b)]}
        bt = ins.better(k, a, b)
        row[f"{A.name} vs {cmp_name}"] = "-" if bt is None else ("✅ better" if bt else "⚠️ weaker")
        row["Formula"] = m["formula"]
        rows.append(row)
    return pd.DataFrame(rows)


def history_table(groups):
    keys = [k for k, m in mt.M.items() if m["group"] in groups]
    years = sorted(set(mA.columns) | set(cmp_df.columns))
    rows = []
    for k in keys:
        for nm, df in ((A.name, mA), (cmp_name, cmp_df)):
            if k not in df.index:
                continue
            r = {"Metric": mt.M[k]["label"], "Entity": nm}
            for y in years:
                r[f"FY{y}"] = fmtv(k, df.loc[k, y]) if y in df.columns else "n/a"
            rows.append(r)
    return pd.DataFrame(rows)


def trend(key, ser: dict, h=260):
    m = mt.M[key]
    fig = go.Figure()
    for i, (nm, s) in enumerate(ser.items()):
        s = s.dropna()
        if s.empty:
            continue
        bench = (i == 1 and B is None)
        fig.add_trace(go.Scatter(x=[f"FY{y}" for y in s.index], y=s.values, name=nm, mode="lines+markers",
                                 line=dict(color=CB if bench else (C1 if i == 0 else C2), dash="dash" if bench else "solid", width=2.5),
                                 marker=dict(size=7)))
    if m["dir"] is not None and key != "fcf":
        for val, col in ((m["green"], "#2da44e"), (m["red"], "#cf222e")):
            if val is not None:
                fig.add_hline(y=val, line=dict(color=col, width=1, dash="dot"))
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), title=dict(text=m["label"], font=dict(size=14)),
                      legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                      yaxis=dict(tickformat=".0%" if m["unit"] == "%" else None), xaxis=dict(type="category"),
                      template="plotly_white")
    return fig


def charts(groups, tag):
    keys = [k for k, v in mt.M.items() if v["group"] in groups]
    shown, missing = [], []
    for k in keys:
        s = {n: df.loc[k] for n, df in series.items() if k in df.index}
        if any(v.notna().any() for v in s.values()):
            shown.append((k, s))
        else:
            missing.append(mt.M[k]["label"])
    cols = st.columns(2)
    for i, (k, s) in enumerate(shown):
        with cols[i % 2]:
            st.plotly_chart(trend(k, s), width="stretch", key=f"{tag}-{k}")
            st.caption(f"`{mt.M[k]['formula']}`")
    if missing:
        st.info("Not available from the data source for this selection: " + ", ".join(missing))


def section(groups, tag, intro=None):
    """Comparison table + year-by-year history + charts for one or more metric groups."""
    if intro:
        st.caption(intro)
    st.markdown("##### Comparison (latest fiscal year)")
    st.dataframe(latest_table(groups), width="stretch", hide_index=True)
    st.caption("🟢 healthy · 🟡 watch · 🔴 concern. Thresholds for every metric are listed in *Data & method*.")
    with st.expander("Year-by-year values", expanded=False):
        st.dataframe(history_table(groups), width="stretch", hide_index=True)
    st.markdown("##### Trends")
    charts(groups, tag)


# ------------------------------------------------------------------ header
if A.currency and A.currency != "INR":
    st.warning(f"Yahoo reports {A.name}'s financials in {A.currency}; the dashboard assumes INR crore.")
st.title(f"{A.name} vs {cmp_name}")
st.caption(f"{sector} · {'Peer company review' if B else 'Individual stats vs industry ' + how.lower()} · "
           f"latest fiscal year in data: FY{yearA} · figures in Rs crore · source: Yahoo Finance via yfinance")


def kpi_row(comp, vals, flags, color, label):
    g, a, r = counts(vals, mt.M)
    st.markdown(f"<div class='coname' style='color:{color}'>● {label}</div>", unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("🟢 Green", g); c[1].metric("🟡 Amber", a); c[2].metric("🔴 Red", r)
    c[3].metric("Red-flag alerts", len(flags))
    rv = comp.fin["rev"].iloc[-1]
    c[4].metric("Revenue (Rs lakh cr)" if rv >= 1e5 else "Revenue (Rs cr)", f"{rv/1e5:,.2f}" if rv >= 1e5 else f"{rv:,.0f}")
    pat = comp.fin["ni"].iloc[-1] if "ni" in comp.fin else np.nan
    c[5].metric("Net profit (Rs cr)", "n/a" if pd.isna(pat) else f"{pat:,.0f}")


kpi_row(A, vA, flagsA, C1, A.name)
if B is not None:
    kpi_row(B, vB, flagsB, C2, B.name)
else:
    g, a, r = counts(vB, mt.M)
    st.markdown(f"<div class='coname' style='color:{CB}'>● {cmp_name} ({len(peers)} peers)</div>", unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("🟢 Green", g); c[1].metric("🟡 Amber", a); c[2].metric("🔴 Red", r)
    rv = np.nanmedian([p.fin["rev"].iloc[-1] for t, p in companies.items() if t != t1])
    c[3].metric("Median revenue (Rs lakh cr)" if rv >= 1e5 else "Median revenue (Rs cr)",
                f"{rv/1e5:,.2f}" if rv >= 1e5 else f"{rv:,.0f}")

tabs = st.tabs(["Overview", "Liquidity", "Solvency & debt service", "Earnings quality", "Distress & forensic scores",
                "Insights & analysis", "Data & method"])

# ------------------------------------------------------------------ Overview: one rectangular tab per umbrella factor
with tabs[0]:
    st.caption("Pick a factor to see every metric in it for the selected company and comparator.")
    sub = st.tabs(mt.GROUPS)
    for gname, t in zip(mt.GROUPS, sub):
        with t:
            gA = counts(vA, [k for k, m in mt.M.items() if m["group"] == gname])
            gB = counts(vB, [k for k, m in mt.M.items() if m["group"] == gname])
            st.markdown(f"**{gname}** - {GROUP_TITLES[gname]}")
            st.write(f"{A.name}: 🟢 {gA[0]} · 🟡 {gA[1]} · 🔴 {gA[2]}   |   {cmp_name}: 🟢 {gB[0]} · 🟡 {gB[1]} · 🔴 {gB[2]}")
            st.dataframe(latest_table([gname]), width="stretch", hide_index=True)
            st.caption("🟢 healthy · 🟡 watch · 🔴 concern. ‘-’ means informational (no threshold) or not available.")

with tabs[1]:
    st.subheader("Liquidity")
    section(["Liquidity"], "liq", GROUP_TITLES["Liquidity"])
with tabs[2]:
    st.subheader("Solvency & debt service")
    section(["Solvency", "Returns"], "sol", f"{GROUP_TITLES['Solvency']} {GROUP_TITLES['Returns']}")
    cr = vA.get("cce", np.nan); dr = vA.get("debt_ratio", np.nan)
    if not pd.isna(cr) and not pd.isna(dr):
        cash_rich, asset_rich = cr >= 0.6, dr <= 0.5
        approach = {(True, True): "debt sculpting", (True, False): "debt avalanche", (False, True): "debt sizing (DSCR-based)",
                    (False, False): "debt snowball"}[(cash_rich, asset_rich)]
        st.markdown("##### Debt structuring view")
        st.write(f"{A.name} screens as **{'cash-rich' if cash_rich else 'cash-poor'} + {'asset-rich' if asset_rich else 'asset-poor'}** "
                 f"(proxy: CFO/EBITDA ≥ 60% and Debt/Assets ≤ 50%) → suggested approach: **{approach}**.")
with tabs[3]:
    st.subheader("Earnings quality")
    section(["Earnings quality", "Market"], "eq", f"{GROUP_TITLES['Earnings quality']} P/OCF above P/E means profits are not converting to cash.")
    st.markdown("**Reading the accrual ratios:** low balance-sheet accrual + low cash-flow accrual = excellent quality; both high = serious concern; "
                "if only one is high, investigate that side.")
with tabs[4]:
    st.subheader("Distress & forensic scores")
    section(["Scores"], "sc", GROUP_TITLES["Scores"])
    st.markdown("##### Benford's-law screen")
    cols = st.columns(2 if B is not None else 1)
    for col, comp, r, colr in zip(cols, (A, B), (bfA, bfB), (C1, C2)):
        if comp is None:
            continue
        with col:
            st.markdown(f"**{comp.name}**")
            if not r or r["actual"] is None:
                st.info("Too few reported figures for a Benford test (needs ≥ 30).")
                continue
            fig = go.Figure()
            fig.add_bar(x=list(range(1, 10)), y=r["actual"], name="Actual", marker_color=colr)
            fig.add_scatter(x=list(range(1, 10)), y=r["expected"], name="Expected (Benford)", mode="lines+markers",
                            line=dict(color=CB, dash="dash"))
            fig.update_layout(height=280, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(tickformat=".0%"), xaxis=dict(title="Leading digit", dtick=1),
                              legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, width="stretch", key=f"bf-{comp.ticker}")
            st.write(f"KS = **{r['ks']:.3f}** vs cut-off 1.36/√N = **{r['cutoff']:.3f}** (N = {r['n']}) → "
                     + ("🔴 above cut-off: unusual pattern, investigate" if r["flag"] else "🟢 within cut-off"))
    st.caption("Benford's law is an anomaly detector, not a fraud detector: it shows where to look.")
    st.markdown("##### Governance (Yahoo/ISS risk scores, 1 = low risk, 10 = high risk)")
    gv = {c.name: c.gov for c in (A, B) if c is not None}
    gdf = pd.DataFrame(gv).reindex(["auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk"])
    gdf.index = ["Audit risk", "Board risk", "Compensation risk", "Shareholder-rights risk", "Overall risk"]
    st.dataframe(gdf, width="stretch")
    with st.expander("Manual governance & related-party checklist (not available through any free API)"):
        st.markdown("""
- **Related-party advances ÷ revenue** - normally 2-10%; above 50% is a red flag.
- **Purchases booked ÷ increase in inventory** - a very high multiple means money is not turning into goods.
- **Promoter pledging; company assets pledged for group-company loans.**
- **Quality of funding**: is fresh equity really fresh money? Bank trail, common directors/addresses.
- **Revenue reality**: debtor days, customer concentration, GST/customer verification.
- **Auditor**: Big-4 or not, tenure, non-audit fees vs audit fees, opinion type (unqualified / qualified / adverse / disclaimer).
- **Board**: CEO duality, independent-director share, CEO tenure/turnover.""")

# ------------------------------------------------------------------ Insights
with tabs[5]:
    st.subheader("Insights & analysis")

    def render(secs):
        for head, bullets in secs:
            st.markdown(f"#### {head}")
            for b in bullets:
                st.markdown(f"- {b}")

    if B is None:
        render(ins.build(A.name, vA, flagsA, cmp_label, vB, None, bfA))
    else:
        l, r = st.columns(2)
        with l:
            st.markdown(f"<div class='coname' style='color:{C1}'>● {A.name}</div>", unsafe_allow_html=True)
            render(ins.build(A.name, vA, flagsA, B.name, vB, B.name, bfA))
        with r:
            st.markdown(f"<div class='coname' style='color:{C2}'>● {B.name}</div>", unsafe_allow_html=True)
            render(ins.build(B.name, vB, flagsB, A.name, vA, A.name, bfB))

# ------------------------------------------------------------------ Data & method
with tabs[6]:
    st.subheader("Data used")
    for comp in ([A] + ([B] if B else [])):
        st.write(f"Standardised annual statements for **{comp.name}** (Rs crore):")
        st.dataframe(comp.fin.T.style.format("{:,.1f}"), width="stretch")
        if comp.missing:
            st.warning(f"Fields Yahoo did not supply for {comp.name} (related metrics show n/a or use fallbacks): " + ", ".join(comp.missing))
    st.subheader("Metric dictionary and thresholds")
    st.dataframe(pd.DataFrame([{"Metric": m["label"], "Group": m["group"], "Formula": m["formula"],
                                "Better when": {"high": "higher", "low": "lower", None: "-"}[m["dir"]],
                                "Green at": m["green"], "Red at": m["red"]} for m in mt.M.values()]),
                 width="stretch", hide_index=True)
    st.subheader("Assumptions & limitations")
    st.markdown("""
- **Yahoo Finance gives about 4 annual periods**, so averages, Piotroski, Beneish and the Ohlson change term need the prior year and start in year 2.
- **Fiscal-year label** = calendar year in which the period ends (FY2025 = year ending Mar-2025 for most Indian firms).
- **DSCR is a proxy**: (CFO + interest) ÷ (interest + current portion of debt). True DSCR needs the repayment schedule.
- **Defensive interval** uses cash opex = Revenue − EBITDA; liquid assets = cash & short-term investments + receivables.
- **Cash-flow accrual** = (NI − CFO) ÷ average net operating assets; NOA = (Assets − cash) − (Liabilities − debt).
- **Altman Z (original)** uses *current* market cap, so only the latest year is shown. Z′ and Z″ use book values.
- **Beneish**: if SG&A or another index is missing, that index is set to neutral (1.0); results are then indicative. It is not valid for banks/insurers (none are in the sector list).
- **Ohlson** size term depends on units: total assets in Rs crore ÷ price index (default 100).
- **Not included**: Montier C-score, Kaplan-Zingales, Fulmer H, abnormal-accrual regressions, LLCR and time-adjusted current ratio (these need repayment schedules or receivable ageing).
- Industry average = the companies listed for that sector in `sectors.py`, not the whole market.
- Insights are rule-based text, not a credit rating.""")
