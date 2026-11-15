import warnings
from html import escape as esc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dataclasses
from datetime import datetime, timezone
import importlib
import os

import data as dt
import early_warning as ew
import insights as ins
import report as rp
import metrics as mt
import news as nw
import snapshot as sn
import sectors


def _fresh(mod):
    """Streamlit re-runs app.py after a code update (e.g. a git push to Streamlit Cloud) but keeps imported modules in
    memory, so a new app.py could run against an old metrics.py. Reload a module only when its file has changed, so
    concurrent sessions are not disturbed. Reloading data.py also empties its download cache, so after a deploy that
    changes data.py the next visitor re-downloads the sector once."""
    stamp = os.path.getmtime(mod.__file__)
    if getattr(mod, "_loaded_mtime", None) != stamp:
        mod = importlib.reload(mod)
        mod._loaded_mtime = stamp
    return mod


dt = _fresh(dt)
mt = _fresh(mt)          # metrics before insights: insights depends on it
ins = _fresh(ins)
ew = _fresh(ew)
rp = _fresh(rp)
nw = _fresh(nw)
sn = _fresh(sn)
SECTORS = _fresh(sectors).SECTORS

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Credit Risk Analyst Dashboard", layout="wide", page_icon="📊")

C1, C2, CB = "#1f6feb", "#e8710a", "#8b949e"          # company 1, company 2, benchmark
GREEN, AMBER, RED = "#2da44e", "#d29922", "#cf222e"
SIG = {"G": "✓ Healthy", "A": "! Watch", "R": "✕ Concern", None: "–"}
SIG_CSS = {"✓ Healthy": "background-color: rgba(46,160,67,.30)", "! Watch": "background-color: rgba(210,153,34,.32)",
           "✕ Concern": "background-color: rgba(248,81,73,.32)"}
SIGNAL_COL = {"ELEVATED CONCERN": RED, "WATCH": AMBER, "COMFORTABLE": GREEN}
GROUP_TITLES = {
    "Liquidity": "Can the borrower pay short-term dues from real cash?",
    "Solvency": "Can it carry its debt?",
    "Earnings quality": "Is profit backed by cash?",
    "Returns": "Are returns above the cost of capital?",
    "Growth & margins": "How fast is it growing and how much of each rupee of sales does it keep? (size-neutral)",
    "Scores": "Distress and manipulation scores",
    "Market": "What does the market price imply?",
}
ADV = '<span class="adv">ADVANCED</span>'

st.markdown("""<style>
.block-container{padding-top:3rem;max-width:1450px}
.adv{display:inline-block;background:#6e40c9;color:#fff;font-size:.66rem;font-weight:700;letter-spacing:.07em;
     padding:2px 9px;border-radius:999px;vertical-align:middle;margin-left:.5rem}
.banner{border-radius:12px;padding:14px 18px;border-left:8px solid var(--c);background:rgba(128,128,128,.10);margin-bottom:.6rem}
.banner .sig{font-size:1.4rem;font-weight:800;color:var(--c);line-height:1.2}
.banner .who{font-size:.85rem;opacity:.75;font-weight:600}
.card{border:1px solid rgba(128,128,128,.30);border-radius:12px;padding:12px 14px;height:100%;background:rgba(128,128,128,.05)}
.card .t{font-weight:700;margin-bottom:2px}
.bar{height:8px;border-radius:99px;background:rgba(128,128,128,.25);overflow:hidden;margin:3px 0 6px 0}
.bar>div{height:100%;border-radius:99px}
.small{font-size:.82rem;opacity:.78}
div[data-testid="stMetric"]{background:rgba(128,128,128,.09);padding:8px 12px;border-radius:10px}
.coname{font-weight:700;font-size:1.05rem;margin:.1rem 0 .3rem 0}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
fmtv = ins.fmt


def counts(vals, keys):
    r = [mt.rag(k, vals.get(k)) for k in keys]
    return r.count("G"), r.count("A"), r.count("R")


def score_color(s):
    return GREEN if s >= 65 else AMBER if s >= 40 else RED


def bench_frame(peer_metrics: dict, how: str):
    frames = [m.reindex(list(mt.M)) for m in peer_metrics.values()]
    years = sorted(set().union(*[f.columns for f in frames]))
    arr = np.stack([f.reindex(columns=years).values.astype(float) for f in frames])
    fn = np.nanmedian if how == "Median" else np.nanmean
    with np.errstate(all="ignore"):
        res = fn(arr, axis=0)
    out = pd.DataFrame(res, index=list(mt.M), columns=years)
    out.loc[SIZE_KEYS] = np.nan            # a median of rupee amounts mixes big and small firms: not a benchmark
    return out


SIZE_KEYS = [k for k, m in mt.M.items() if m["unit"] == "Rs cr"]
NOT_COMPARABLE = "not comparable (size)"


def style_cells(df, css_fn, subset):
    sty = df.style
    return (sty.map if hasattr(sty, "map") else sty.applymap)(css_fn, subset=subset)


def money(v):
    if v is None or pd.isna(v):
        return "n/a"
    return f"{v/1e5:,.2f}L" if abs(v) >= 1e5 else f"{v:,.0f}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 Credit Risk Dashboard")
st.sidebar.caption("For the **Credit Risk Analyst** · live data from Yahoo Finance")
view_box = st.sidebar.expander("View options", expanded=True)
with view_box:
    basis_sel = st.segmented_control(
        "Scoring basis", ["Textbook", "Sector-relative"], default="Textbook", key="basis",
        help="Textbook: the course cut-offs, the same for every industry. Sector-relative: each metric is judged against the "
             "companies in the selected sector - healthy = at least the sector median, concern = the sector's worst 25%. "
             "Hard floors still apply (DSCR or interest cover below 1x, Ohlson default probability above 50%), and the "
             "Beneish and free-cash-flow rules never become 'sector-normal'.") or "Textbook"
    strict = st.segmented_control("Threshold strictness", ["Lenient", "Standard", "Conservative"], default="Standard",
                                  key="strict", help="Textbook basis: Conservative moves every healthy / concern cut-off 20% towards "
                                  "'better', Lenient 20% the other way. Sector-relative basis: Conservative uses the 60th / 35th sector "
                                  "percentiles, Standard the 50th / 25th, Lenient the 40th / 15th. Model cut-offs (Beneish −2.22) never move.") or "Standard"
    show_calc = st.toggle("Show calculations", key="show_calc",
                          help="Adds the formula with the company's own numbers to every comparison table.")
mt.set_strictness(strict)
with st.sidebar.expander("Settings", expanded=False):
    how = st.radio("Industry average =", ["Median", "Mean"], horizontal=True)
    gnp = st.number_input("Price index for size term (Ohlson)", 1.0, 1000.0, 100.0,
                          help="Total assets (Rs crore) are divided by this index in the Ohlson size term.")
if st.sidebar.button("Reload data", help="Skip the nightly snapshot and fetch everything again from Yahoo Finance and Google News"):
    dt.clear_cache()
    nw.clear_cache()
    st.rerun()
_src = dt.source()
if _src["mode"] == "snapshot":
    st.sidebar.caption(f"Data: nightly snapshot of {_src['built'].astimezone(dt.IST):%d %b, %H:%M IST} "
                       f"({_src['age_h']:.0f} h old). *Reload data* fetches live.")
else:
    st.sidebar.caption("Data: live from Yahoo Finance" + (f" (last snapshot {_src['built'].astimezone(dt.IST):%d %b} is too old to use)"
                                                          if _src["built"] else "") + ".")
st.sidebar.markdown("---")
st.sidebar.markdown("**How to read signals**  \n✓ Healthy · ! Watch · ✕ Concern  \n"
                    f"{ADV} marks analysis beyond the core ratios.", unsafe_allow_html=True)

# ------------------------------------------------------------------ top selector bar (steps 1-3)
st.markdown("### Credit Risk Analyst Dashboard")
top = st.columns([1.1, 1.2, 1.35, 1.35])
with top[0]:
    sector = st.selectbox("1 · Sector", list(SECTORS))
with top[1]:
    mode = st.selectbox("2 · Analysis type", ["Individual vs industry average", "Peer company review"])
individual = mode.startswith("Individual")

tickers = SECTORS[sector]
missing_t = [t for t in tickers if not dt.is_cached(t)]
if missing_t:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    bar = st.progress(0.0, text=f"Loading {len(missing_t)} companies from Yahoo Finance…")
    probe = missing_t[:3]
    if not any(dt.is_cached(t) for t in tickers) and all(dt.get_company(t, tickers[t])[0] is None for t in probe):
        for t2 in missing_t[3:]:            # Yahoo unreachable: stop early instead of waiting for every ticker
            dt.mark_failed(t2, "skipped - Yahoo Finance unreachable (first 3 requests failed)")
    else:
        todo = [t for t in missing_t if not dt.is_cached(t)]
        with ThreadPoolExecutor(max_workers=6) as pool:      # a few parallel requests; more risks Yahoo rate limits
            futs = {pool.submit(dt.get_company, t, tickers[t]): t for t in todo}
            for i, f in enumerate(as_completed(futs), 1):
                bar.progress(i / len(todo), text=f"Loaded {tickers[futs[f]]} ({i} of {len(todo)})…")
    bar.empty()

companies, errors = {}, {}
for t, n in tickers.items():
    c, e = dt.get_company(t, n)            # served from cache
    if c is not None and not c.fin.empty:
        companies[t] = c
    else:
        errors[t] = e or "no usable data"
# ---- as-of year: cut every company's statements at the chosen fiscal year
all_years = sorted(set().union(*[set(c.fin.index) for c in companies.values()])) if companies else []
year_opts = [y for y in all_years if sum(y in c.fin.index for c in companies.values()) >= max(3, len(companies) // 2)]
with view_box:
    asof_y = st.selectbox("As-of fiscal year", year_opts[::-1], format_func=lambda y: f"FY{y}" + (" (latest)" if y == year_opts[-1] else ""),
                          key="asof", help="View the whole dashboard as it looked at an earlier year end. Market cap is only "
                          "available today, so market-based measures (Altman Z, P/E, P/OCF) are blank for past years.") if year_opts else None
if asof_y is not None and asof_y != year_opts[-1]:
    cut = {}
    for t, c in companies.items():
        f_ = c.fin.loc[c.fin.index <= asof_y]
        if len(f_):
            cut[t] = dataclasses.replace(c, fin=f_, mcap_cr=np.nan, period_end=f"FY{asof_y}",
                                         raw_values={y: v for y, v in c.raw_values.items() if y <= asof_y})
    companies = cut
    st.sidebar.info(f"Showing the dashboard as of **FY{asof_y}**. Switch back to the latest year in *View options*.")
names = {t: c.name for t, c in companies.items()}
if errors:
    st.sidebar.warning(f"{len(errors)} ticker(s) failed to load: " + ", ".join(tickers[t] for t in errors))
_saved = [c.name for c in companies.values() if "saved copy" in (c.fetched_at or "")]
if _saved:
    st.sidebar.warning(f"Yahoo did not respond for {len(_saved)} compan{'y' if len(_saved) == 1 else 'ies'}; showing the last "
                       "nightly copy: " + ", ".join(_saved))
if not names:
    st.error("No company data could be loaded for this sector.")
    st.write({tickers[t]: e for t, e in errors.items()})
    st.info("Yahoo Finance may be unreachable or rate-limiting. Click **Reload data** in the sidebar and retry in a few minutes.")
    st.stop()

with top[2]:
    t1 = st.selectbox("3 · Company" if individual else "3 · Company 1", list(names), format_func=lambda t: names[t])
t2 = None
with top[3]:
    if individual:
        st.markdown("&nbsp;")
    else:
        rest = [t for t in names if t != t1]
        if not rest:
            st.error("Need at least two loaded companies for a peer review.")
            st.stop()
        t2 = st.selectbox("Company 2", rest, format_func=lambda t: names[t])

# ------------------------------------------------------------------ compute
all_m = {t: mt.compute(c.fin, c.mcap_cr, gnp) for t, c in companies.items()}
latest_all = {t: ins.latest(m) for t, m in all_m.items()}
SECTOR_VALS = {k: [lv.get(k) for lv in latest_all.values()] for k in mt.M}
mt.set_basis(basis_sel, SECTOR_VALS)
SECTOR_MODE = basis_sel == "Sector-relative"
if SECTOR_MODE:
    n_band = sum(mt.is_sector_band(k) for k in mt.M)
    st.sidebar.info(f"**Sector-relative scoring:** {n_band} metrics are judged against the {len(latest_all)} {sector} companies. "
                    "Tables add a *textbook* column so you can see what is weak on paper but normal for the sector.")
W = {p: st.session_state.get(f"w_{p}", d) for p, d in mt.DEFAULT_WEIGHTS.items()}

A = companies[t1]; mA = all_m[t1]
yearA = mA.columns[-1]
vA = latest_all[t1]
flagsA = mt.red_flags(A.fin, mA)
if individual:
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
    gov_cmp = np.nanmedian([mt.governance_score(companies[t].gov) for t in peers]) if any(
        not pd.isna(mt.governance_score(companies[t].gov)) for t in peers) else np.nan
    pB = mt.pillar_scores(vB); pB["Governance"] = gov_cmp
else:
    B = companies[t2]; cmp_df = all_m[t2]; cmp_name = B.name; cmp_label = B.name
    vB = latest_all[t2]
    flagsB = mt.red_flags(B.fin, cmp_df)
    pB = mt.pillar_scores(vB, B.gov)
pA = mt.pillar_scores(vA, A.gov)
scoreA, scoreB = mt.overall_score(pA, W), mt.overall_score(pB, W)

series = {A.name: mA, cmp_name: cmp_df.reindex(list(mt.M))}
rawA = [x for y in A.raw_values for x in A.raw_values[y]]
bfA = mt.benford(rawA) if rawA else None
bfB = None
if B is not None:
    rawB = [x for y in B.raw_values for x in B.raw_values[y]]
    bfB = mt.benford(rawB) if rawB else None


# ------------------------------------------------------------------ tables, charts, sections
def calc_text(k, comp, year=None):
    """'inputs = result' for a metric, or the bare formula when it cannot be filled in."""
    ex = mt.explain(k, comp.fin, year)
    m_ = all_m[comp.ticker]
    y = year if year is not None else m_.columns[-1]
    v = m_.at[k, y] if (k in m_.index and y in m_.columns) else np.nan
    return f"{ex} = {fmtv(k, v)}" if ex else mt.M[k]["formula"]


def table_df(groups):
    rows = []
    for k, m in mt.M.items():
        if m["group"] not in groups:
            continue
        a, b = vA.get(k), vB.get(k)
        tv = [float(x) for x in mA.loc[k].values if not pd.isna(x)]
        bt = ins.better(k, a, b)
        pct = mt.percentile(k, a, [lv.get(k) for lv in latest_all.values()])
        size_k = B is None and k in SIZE_KEYS
        row = {"Metric": m["label"], A.name: fmtv(k, a), f"{A.name} signal": SIG[mt.rag(k, a)],
               cmp_name: NOT_COMPARABLE if size_k else fmtv(k, b), f"{cmp_name} signal": SIG[mt.rag(k, b)]}
        if B is None:
            pv = [lv.get(k) for t, lv in latest_all.items() if t != t1 and not pd.isna(lv.get(k))]
            row["Sector middle 50%"] = ("–" if size_k or len(pv) < 4 else
                                        f"{fmtv(k, np.percentile(pv, 25))} to {fmtv(k, np.percentile(pv, 75))}")
        if SECTOR_MODE:
            row[f"{A.name} textbook signal"] = SIG[mt.rag(k, a, "Textbook")]
        if show_calc:
            for comp in (A, B):
                if comp is not None:
                    row[f"{comp.name} calculation"] = calc_text(k, comp)
            if B is None:
                row["Formula"] = m["formula"]
        rows.append({**row,
                     "Edge": "–" if bt is None else ("Better" if bt else "Weaker"),
                     "Sector percentile": None if pd.isna(pct) else float(pct),
                     "Trend": tv if len(tv) >= 2 else None})
    return pd.DataFrame(rows)


def show_table(df, key):
    sig_cols = [c for c in df.columns if c.endswith(" signal")]
    st.dataframe(
        style_cells(df, lambda v: SIG_CSS.get(v, ""), sig_cols),
        column_config={
            "Metric": st.column_config.TextColumn(width="medium"),
            "Sector percentile": st.column_config.ProgressColumn("Sector percentile", min_value=0, max_value=100, format="%.0f",
                                                                 help="Share of companies in this sector that the company beats on this metric. Advanced."),
            "Trend": st.column_config.LineChartColumn("Trend", width="small"),
            "Sector middle 50%": st.column_config.TextColumn(help="Range covering the middle half of the other companies in this sector "
                                                                  "(25th to 75th percentile) - the typical spread around the average."),
        }, hide_index=True, width="stretch", key=key)


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


def trend(key, ser: dict, h=270):
    m = mt.M[key]
    fig = go.Figure()
    allv = []
    for i, (nm, s) in enumerate(ser.items()):
        s = s.dropna()
        if s.empty:
            continue
        allv += list(s.values)
        bench = (i == 1 and B is None)
        fig.add_trace(go.Scatter(x=[f"FY{y}" for y in s.index], y=s.values, name=nm, mode="lines+markers",
                                 line=dict(color=CB if bench else (C1 if i == 0 else C2), dash="dash" if bench else "solid", width=2.6),
                                 marker=dict(size=7)))
    g, r = mt.thresholds(key)
    if m["dir"] is not None and allv and g is not None and r is not None:
        lo, hi = min(allv + [g, r]), max(allv + [g, r])
        pad = (hi - lo) * 0.18 or 1.0
        lo, hi = lo - pad, hi + pad
        zones = ([(lo, r, RED), (r, g, AMBER), (g, hi, GREEN)] if m["dir"] == "high"
                 else [(lo, g, GREEN), (g, r, AMBER), (r, hi, RED)])
        for y0, y1, col in zones:
            if y1 > y0:
                fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.11, line_width=0, layer="below")
        fig.update_yaxes(range=[lo, hi])
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
            need = mt.missing_inputs(k, A.fin)
            missing.append(f"{mt.M[k]['label']} ({'Yahoo has no ' + ', '.join(need) if need else 'needs more history or market data'})")
    cols = st.columns(2)
    for i, (k, s) in enumerate(shown):
        with cols[i % 2]:
            st.plotly_chart(trend(k, s), width="stretch", key=f"{tag}-{k}")
            st.caption(f"{mt.M[k]['desc']}  \n`{mt.M[k]['formula']}`")
    if missing:
        st.info("Not available for this selection: " + "; ".join(missing))


def na_note(groups):
    notes = []
    for k, m in mt.M.items():
        if m["group"] in groups and pd.isna(vA.get(k)):
            need = mt.missing_inputs(k, A.fin)
            notes.append(f"**{m['label']}** - " + ("Yahoo has no " + ", ".join(need) if need else "needs another year of history or market data"))
    if notes:
        with st.expander(f"Why some values show n/a for {A.name} ({len(notes)})"):
            st.markdown("\n".join(f"- {n}" for n in notes))


def section(groups, tag, intro=None):
    if intro:
        st.caption(intro)
    st.markdown("##### Comparison (latest fiscal year)")
    df = table_df(groups)
    show_table(df, f"tbl-{tag}")
    c1, c2 = st.columns([4, 1])
    c1.caption("✓ Healthy · ! Watch · ✕ Concern. Thresholds for every metric are listed in *Data & method*. "
               "Sector percentile is an advanced extension.")
    c2.download_button("Download CSV", df.drop(columns=["Trend"]).to_csv(index=False), file_name=f"{tag}_comparison.csv",
                       mime="text/csv", key=f"dl-{tag}")
    na_note(groups)
    with st.expander("Year-by-year values"):
        st.dataframe(history_table(groups), width="stretch", hide_index=True)
    with st.expander("Metric guide - what each one tells you"):
        for k, m in mt.M.items():
            if m["group"] in groups:
                st.markdown(f"**{m['label']}** - {m['desc']}  \n`{m['formula']}`")
    st.markdown("##### Trends")
    st.caption("Shaded bands: green = healthy, amber = watch, red = concern.")
    charts(groups, tag)


# ------------------------------------------------------------------ headline: verdict banners + KPI strip
def banner(label, vals, flags, score, color_tag):
    v = ins.verdict(vals, flags)
    col = SIGNAL_COL[v["signal"]]
    st.markdown(
        f"<div class='banner' style='--c:{col}'><div class='who' style='color:{color_tag}'>● {esc(label)}</div>"
        f"<div class='sig'>{v['signal']}</div>"
        f"<div>Indicative score <b>{'n/a' if pd.isna(score) else f'{score:.0f}/100'}</b> · {mt.band(score)} · "
        f"{v['g']} healthy · {v['a']} watch · {v['r']} concern</div>"
        f"<div class='small'>Key drivers: {esc(' · '.join(v['reasons']) or 'none flagged')}</div></div>", unsafe_allow_html=True)


def yoy(comp, k):
    """Change in metric k versus the prior year, as delta text for st.metric."""
    m_ = all_m[comp.ticker]
    if k not in m_.index or len(m_.columns) < 2:
        return None
    a, b = m_.at[k, m_.columns[-1]], m_.at[k, m_.columns[-2]]
    if pd.isna(a) or pd.isna(b):
        return None
    return (f"{(a - b) * 100:+.1f} pp vs FY{m_.columns[-2]}" if mt.M[k]["unit"] == "%"
            else f"{a - b:+.2f}x vs FY{m_.columns[-2]}")


def kpi_help(k, comp=None):
    h = mt.M[k]["desc"]
    if comp is not None:
        h += f"\n\n**Calculation (Rs cr):** {calc_text(k, comp)}"
        prev = yoy(comp, k)
        if prev and B is None:
            h += f"\n\n**Change:** {prev}"
    return h


def kpis(comp, vals):
    """Peer-review headline numbers with the change from the prior year."""
    f_ = comp.fin
    rev, ni = f_["rev"].iloc[-1], (f_["ni"].iloc[-1] if "ni" in f_ else np.nan)
    pct = lambda col: (None if len(f_) < 2 or col not in f_ or pd.isna(f_[col].iloc[-2]) or f_[col].iloc[-2] == 0 or pd.isna(f_[col].iloc[-1])
                       else f"{(f_[col].iloc[-1] / f_[col].iloc[-2] - 1):+.1%} vs FY{f_.index[-2]}")
    c = st.columns(3)
    c[0].metric("Revenue (Rs cr)", money(rev), delta=pct("rev"), help=f"L = lakh crore. FY{f_.index[-1]} vs the prior year.")
    c[1].metric("Net profit (Rs cr)", money(ni), delta=pct("ni"), help="L = lakh crore.")
    c[2].metric("Net debt / EBITDA", fmtv("nd_ebitda", vals.get("nd_ebitda")), delta=yoy(comp, "nd_ebitda"), delta_color="inverse",
                help=kpi_help("nd_ebitda", comp))
    c = st.columns(3)
    for col, k, lbl in zip(c, ("icr", "dscr", "cce"), ("Interest cover", "DSCR (proxy)", "Cash conversion")):
        col.metric(lbl, fmtv(k, vals.get(k)), delta=yoy(comp, k), help=kpi_help(k, comp))


REL_KPIS = [[("rev_growth", "Revenue growth"), ("ebitda_margin", "EBITDA margin"), ("net_margin", "Net margin")],
            [("nd_ebitda", "Net debt / EBITDA"), ("icr", "Interest cover"), ("dscr", "DSCR (proxy)"), ("cce", "Cash conversion")]]


def gap_txt(k, a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    d = a - b
    return f"{d * 100:+.1f} pp vs {how.lower()}" if mt.M[k]["unit"] == "%" else f"{d:+.2f}x vs {how.lower()}"


def kpis_rel(vals, bench=None, comp=None):
    """Size-neutral headline numbers; with `bench`, each shows the gap to the industry benchmark."""
    for row in REL_KPIS:
        c = st.columns(len(row))
        for col, (k, lbl) in zip(c, row):
            dl = gap_txt(k, vals.get(k), bench.get(k)) if bench is not None else None
            col.metric(lbl, fmtv(k, vals.get(k)), delta=dl, help=kpi_help(k, comp),
                       delta_color="off" if dl is None else ("normal" if mt.M[k]["dir"] == "high" else "inverse"))


def size_rank(t, field):
    """(rank, of, share of sector total, value) of company t on revenue / net profit / market cap, or None."""
    vals = {x: (c.mcap_cr if field == "mcap" else (c.fin[field].iloc[-1] if field in c.fin else np.nan)) for x, c in companies.items()}
    vals = {x: v for x, v in vals.items() if not pd.isna(v)}
    if t not in vals:
        return None
    order = sorted(vals, key=vals.get, reverse=True)
    pos_total = sum(v for v in vals.values() if v > 0)
    return order.index(t) + 1, len(order), (vals[t] / pos_total if vals[t] > 0 and pos_total else np.nan), vals[t]


def size_line(t):
    parts = []
    for field, lbl in (("rev", "revenue"), ("ni", "net profit"), ("mcap", "market cap")):
        r = size_rank(t, field)
        if r:
            rk, n, sh, v = r
            parts.append(f"{lbl} Rs {money(v)} cr: **#{rk} of {n}**" + (f" ({sh:.0%} of sector)" if field != "ni" and not pd.isna(sh) else ""))
    return "**Size in sector** - " + " · ".join(parts) if parts else ""


st.title(f"{A.name} vs {cmp_name}")
asof = f"Annual statements to {A.period_end or 'n/a'}" + (f" ({B.name}: {B.period_end})" if B is not None and B.period_end != A.period_end else "")
st.caption(f"{sector} · {'Peer company review' if B else 'Individual vs industry ' + how.lower()} · "
           f"scoring: **{basis_sel.lower()}**{'' if strict == 'Standard' else ', ' + strict.lower()} · {asof} · "
           f"fetched {A.fetched_at or 'n/a'} · figures in Rs crore · source: Yahoo Finance (market cap as of fetch)")
if A.currency and A.currency != "INR":
    st.warning(f"Yahoo reports {A.name}'s financials in {A.currency}; the dashboard assumes INR crore.")

hc = st.columns(2)
with hc[0]:
    with st.container(border=True):
        banner(A.name, vA, flagsA, scoreA, C1)
        if B is not None:
            kpis(A, vA)
        else:
            st.caption(size_line(t1))
            kpis_rel(vA, vB, A)
with hc[1]:
    with st.container(border=True):
        if B is not None:
            banner(B.name, vB, flagsB, scoreB, C2)
            kpis(B, vB)
        else:
            banner(f"{cmp_name} ({len(peers)} peers)", vB, [], scoreB, CB)
            st.markdown(f"<div class='small'>{how} of the {len(peers)} other {esc(sector)} companies. Rupee amounts (revenue, profit, "
                        "free cash flow) are ranked, not averaged - see <i>Where it sits in the sector</i> on the Overview tab.</div>",
                        unsafe_allow_html=True)
            kpis_rel(vB)

# ------------------------------------------------------------------ early-warning helpers (used by the tab and the credit memo)
EW_LATEST = asof_y is None or asof_y == year_opts[-1]
fpd = lambda p_: "n/a" if p_ is None or pd.isna(p_) else ("< 0.01%" if p_ < 1e-4 else f"{p_:.2%}")


annual_end_of = ew.annual_end_of
dp_for = ew.dp_for


def ew_bundle(comp, rf):
    """Everything the early-warning view needs for one company (prices for the whole sector, cached 6 hours)."""
    news_ = nw.get_news(comp.name, comp.ticker)
    b_ = ew.bundle(comp, companies, dt.get_prices(list(companies)), dt.get_quarterly(comp.ticker), rf, news_["items"])
    b_["news"] = news_
    return b_


tabs = st.tabs(["Overview", "Liquidity", "Solvency & debt service", "Earnings quality", "Distress & forensic",
                "Early warning", "News & ratings", "Cash flow & lifecycle", "Credit workbench · ADVANCED", "Insights", "Data & method"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    _hist = sn.history()
    _chg = sn.changes(_hist, companies) if len(_hist) else pd.DataFrame()
    if len(_hist) and _hist["date"].nunique() >= 2:
        _worse = int((_chg["Direction"] == "worse").sum()) if len(_chg) else 0
        _lbl = (f"What changed in {sector} since {_chg.attrs['since']:%d %b}: " +
                (f"{len(_chg)} change(s), {_worse} for the worse" if len(_chg) else "nothing"))
        with st.expander(_lbl, expanded=_worse > 0):
            if len(_chg):
                st.dataframe(style_cells(_chg.drop(columns=["ticker"]), lambda v: {"worse": SIG_CSS["✕ Concern"], "better": SIG_CSS["✓ Healthy"]}.get(v, ""),
                                         ["Direction"]), hide_index=True, width="stretch")
            st.caption(f"Compares the last two nightly snapshots ({_chg.attrs['since']:%d %b} → {_chg.attrs['until']:%d %b}) on "
                       "textbook cut-offs and default weights: grade, headline signal, early-warning level, news flow, and score moves of 5+ points.")
            _t = _hist[_hist["ticker"] == t1].sort_values("date")
            if len(_t) >= 2:
                _f = go.Figure(go.Scatter(x=_t["date"], y=_t["score"], mode="lines+markers", line=dict(color=C1, width=2.4),
                                          text=_t["grade"], hovertemplate="%{x|%d %b}: score %{y:.0f} · %{text}<extra></extra>"))
                _f.update_layout(height=200, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                                 title=dict(text=f"{names[t1]}: score by night", font=dict(size=13)), yaxis=dict(range=[0, 100]))
                st.plotly_chart(_f, width="stretch", key="hist-score")
    st.markdown("##### Scorecard by factor")
    st.caption("Each factor scores 0-100 from its metrics (healthy = 100, watch = 50, concern = 0). "
               "Weights and the overall score are an advanced extension (see Credit workbench).")
    PILLAR_NOTE_GROUPS = {"Liquidity": ["Liquidity"], "Solvency & coverage": ["Solvency"], "Earnings quality": ["Earnings quality"],
                          "Returns": ["Returns"], "Distress & forensic": ["Scores"]}

    def worst_note(p):
        if p == "Governance":
            return "Yahoo/ISS risk scores (1 low - 10 high)" if not pd.isna(pA.get("Governance")) else "No governance data from Yahoo"
        groups = PILLAR_NOTE_GROUPS[p]
        reds = [k for k, m in mt.M.items() if m["group"] in groups and mt.rag(k, vA.get(k)) == "R"]
        ambers = [k for k, m in mt.M.items() if m["group"] in groups and mt.rag(k, vA.get(k)) == "A"]
        if reds:
            k = reds[0]; return f"Concern: {mt.M[k]['label']} {fmtv(k, vA[k])}"
        if ambers:
            k = ambers[0]; return f"Watch: {mt.M[k]['label']} {fmtv(k, vA[k])}"
        return "All rated metrics healthy"

    def card(p):
        sa, sb = pA.get(p), pB.get(p)
        def row(nm, s, col):
            if s is None or pd.isna(s):
                return f"<div class='small'>{esc(nm)}: n/a</div>"
            return (f"<div class='small'>{esc(nm)} <b>{s:.0f}</b></div>"
                    f"<div class='bar'><div style='width:{s:.0f}%;background:{score_color(s)}'></div></div>")
        return (f"<div class='card'><div class='t'>{p}</div>{row(A.name, sa, C1)}{row(cmp_name, sb, CB)}"
                f"<div class='small'>{esc(worst_note(p))}</div></div>")

    def cutoff(k, which):
        g, r = mt.thresholds(k)
        v = g if which == "G" else r
        if v is None or mt.M[k]["dir"] is None:
            return "–"
        if k in ("beneish", "fcf") or (k == "springate" and not mt.is_sector_band(k)):
            return {"beneish": {"G": "< −2.22", "R": "> −2.22"}, "springate": {"G": "> 0.862", "R": "< 0.862"},
                    "fcf": {"G": "> 0", "R": "< 0"}}[k][which]
        hi = mt.M[k]["dir"] == "high"
        return (("≥ " if hi else "≤ ") if which == "G" else ("< " if hi else "> ")) + fmtv(k, v)

    def factor_detail(p):
        """Contents of a scorecard box's popover: how the factor score is built, metric by metric."""
        if p == "Governance":
            gv = {c.name: c.gov for c in (A, B) if c is not None}
            gdf = pd.DataFrame(gv).reindex(["auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk"])
            gdf.index = ["Audit risk", "Board risk", "Compensation risk", "Shareholder-rights risk", "Overall risk"]
            st.markdown("**Governance: Yahoo / ISS risk scores** (1 = low risk, 10 = high risk)")
            st.dataframe(gdf, width="stretch")
            st.caption("Factor score = 100 × (10 − average risk score) ÷ 9. Not available for every company.")
            return
        keys = [k for k, m in mt.M.items() if m["group"] in PILLAR_NOTE_GROUPS[p]]
        rg = [mt.rag(k, vA.get(k)) for k in keys]
        nG, nA, nR = rg.count("G"), rg.count("A"), rg.count("R")
        sc = pA.get(p)
        st.markdown(f"**{p}: how the score is built**")
        st.markdown(f"{A.name}: ({nG} healthy × 100 + {nA} watch × 50 + {nR} concern × 0) ÷ {nG + nA + nR} rated metrics = "
                    f"**{'n/a' if sc is None or pd.isna(sc) else f'{sc:.0f}'}**" + (f" · threshold strictness: *{strict}*" if strict != "Standard" else ""))
        rows = []
        for k in keys:
            a, b = vA.get(k), vB.get(k)
            size_k = B is None and k in SIZE_KEYS
            pct = mt.percentile(k, a, [lv.get(k) for lv in latest_all.values()])
            tv = [float(x) for x in mA.loc[k].values if not pd.isna(x)] if k in mA.index else []
            rows.append({"Metric": mt.M[k]["label"], A.name: fmtv(k, a), "Signal": SIG[mt.rag(k, a)],
                         **({"Textbook signal": SIG[mt.rag(k, a, "Textbook")]} if SECTOR_MODE else {}),
                         cmp_name: NOT_COMPARABLE if size_k else fmtv(k, b),
                         "Healthy": cutoff(k, "G"), "Concern": cutoff(k, "R"),
                         "Sector pct": None if pd.isna(pct) else float(pct), "Trend": tv if len(tv) >= 2 else None,
                         f"How {A.name}'s value is calculated": calc_text(k, A)})
        ddf = pd.DataFrame(rows)
        st.dataframe(style_cells(ddf, lambda v: SIG_CSS.get(v, ""), [c for c in ddf.columns if c.endswith("Signal")]),
                     hide_index=True, width="stretch",
                     column_config={"Sector pct": st.column_config.ProgressColumn("Sector pct", min_value=0, max_value=100, format="%.0f"),
                                    "Trend": st.column_config.LineChartColumn("Trend", width="small"),
                                    f"How {A.name}'s value is calculated": st.column_config.TextColumn(width="large")})
        st.caption("Figures in Rs crore. Metrics shown with '–' cut-offs are informational and do not count towards the score."
                   + (f" Cut-offs are {sector}'s own percentiles (sector-relative basis)." if SECTOR_MODE else ""))

    pill = list(mt.PILLARS) + ["Governance"]
    st.caption("Click **See inside** under any box for the metrics, cut-offs and calculations behind its score.")
    for r0 in (0, 3):
        cc = st.columns(3)
        for i, p in enumerate(pill[r0:r0 + 3]):
            with cc[i]:
                st.markdown(card(p), unsafe_allow_html=True)
                with st.popover("See inside", icon=":material/search:", width="stretch", key=f"pop-{p}"):
                    factor_detail(p)
    st.markdown("")
    # ---------- sector position: size is ranked, ratios are shown against the whole spread of peers
    st.markdown("##### Where it sits in the sector")
    st.caption(f"All {len(companies)} {sector} companies loaded. Left: size, which is ranked rather than averaged. Right: any size-neutral "
               "metric, with the sector's middle 50% shaded and the median dashed, so you see the spread, not just one average.")
    HL = {A.name: C1, **({B.name: C2} if B is not None else {})}
    sl, sr = st.columns(2)
    with sl:
        sz = st.segmented_control("Size measure", ["Revenue", "Net profit", "Market cap"], default="Revenue", key="sz-measure") or "Revenue"
        fld = {"Revenue": "rev", "Net profit": "ni", "Market cap": "mcap"}[sz]
        sv = sorted([(c.name, c.mcap_cr if fld == "mcap" else (c.fin[fld].iloc[-1] if fld in c.fin else np.nan))
                     for c in companies.values()], key=lambda x: -np.nan_to_num(x[1], nan=-1e18))
        sv = [x for x in sv if not pd.isna(x[1])]
        sfig = go.Figure(go.Bar(x=[v for _, v in sv], y=[n for n, _ in sv], orientation="h",
                                marker_color=[HL.get(n, "rgba(139,148,158,.45)") for n, _ in sv],
                                text=[f"#{i + 1}" for i in range(len(sv))], textposition="outside",
                                hovertemplate="%{y}: Rs %{x:,.0f} cr<extra></extra>"))
        sfig.update_layout(height=max(320, 22 * len(sv)), template="plotly_white", margin=dict(l=10, r=30, t=10, b=10),
                           yaxis=dict(autorange="reversed"), xaxis_title=f"{sz} (Rs crore)")
        st.plotly_chart(sfig, width="stretch", key="size-rank")
    with sr:
        d_opts = [k for k, m in mt.M.items() if m["dir"] is not None and m["unit"] != "Rs cr"]
        mk = st.selectbox("Metric", d_opts, index=d_opts.index("ebitda_margin"), format_func=lambda k: mt.M[k]["label"], key="dist-metric")
        pts = [(companies[t].name, latest_all[t].get(mk)) for t in companies if not pd.isna(latest_all[t].get(mk))]
        if len(pts) < 3:
            st.info("Too few companies have this metric.")
        else:
            good_first = (lambda v: -v) if mt.M[mk]["dir"] == "high" else (lambda v: v)
            pts.sort(key=lambda x: good_first(x[1]))
            vs = np.array([v for _, v in pts], dtype=float)
            lo, hi = np.percentile(vs, 5), np.percentile(vs, 95)
            for nm in HL:                                   # always keep the highlighted companies on scale
                v_ = dict(pts).get(nm)
                if v_ is not None:
                    lo, hi = min(lo, v_), max(hi, v_)
            pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1.0
            lo, hi = lo - pad, hi + pad
            xs = np.clip(vs, lo, hi)
            pct_ax = mt.M[mk]["unit"] == "%"
            dfig = go.Figure()
            q1, med, q3 = np.percentile(vs, [25, 50, 75])
            dfig.add_vrect(x0=q1, x1=q3, fillcolor=CB, opacity=0.12, line_width=0)
            dfig.add_vline(x=med, line_dash="dash", line_color=CB, annotation_text="median", annotation_position="top")
            for thr, colr in zip(mt.thresholds(mk), (GREEN, RED)):
                if thr is not None and lo < thr < hi:
                    dfig.add_vline(x=thr, line_color=colr, line_width=1, opacity=0.6)
            dfig.add_scatter(x=xs, y=[n for n, _ in pts], mode="markers",
                             marker=dict(size=[14 if n in HL else 9 for n, _ in pts], color=[HL.get(n, "rgba(139,148,158,.75)") for n, _ in pts],
                                         symbol=["diamond" if (v < lo or v > hi) else "circle" for v in vs]),
                             customdata=[fmtv(mk, v) for v in vs], hovertemplate="%{y}: %{customdata}<extra></extra>")
            dfig.update_layout(height=max(320, 22 * len(pts)), template="plotly_white", margin=dict(l=10, r=10, t=24, b=10),
                               yaxis=dict(autorange="reversed"), showlegend=False,
                               xaxis=dict(range=[lo, hi], tickformat=".0%" if pct_ax else None, title=mt.M[mk]["label"]))
            st.plotly_chart(dfig, width="stretch", key="dist-plot")
            rk = [n for n, _ in pts].index(A.name) + 1 if A.name in dict(pts) else None
            st.caption((f"{A.name} ranks **#{rk} of {len(pts)}** (best at the top). " if rk else "")
                       + "Green / red lines: healthy and concern thresholds. Diamonds are outliers pinned to the edge (hover for the real value).")

    st.markdown("##### Pick a factor")
    sub = st.tabs(mt.GROUPS)
    for gname, t in zip(mt.GROUPS, sub):
        with t:
            gA, gB = (counts(v, [k for k, m in mt.M.items() if m["group"] == gname]) for v in (vA, vB))
            st.markdown(f"**{gname}** - {GROUP_TITLES[gname]}")
            st.caption(f"{A.name}: {gA[0]} healthy · {gA[1]} watch · {gA[2]} concern   |   "
                       f"{cmp_name}: {gB[0]} healthy · {gB[1]} watch · {gB[2]} concern")
            d = table_df([gname])
            show_table(d, f"ov-{gname}")

with tabs[1]:
    st.subheader("Liquidity")
    section(["Liquidity"], "liq", GROUP_TITLES["Liquidity"])

with tabs[2]:
    st.subheader("Solvency & debt service")
    section(["Solvency", "Returns", "Growth & margins"], "sol", f"{GROUP_TITLES['Solvency']} {GROUP_TITLES['Returns']}")
    st.markdown("##### Return drivers (DuPont)")
    dpA = mt.dupont(A.fin)
    dpB = mt.dupont(B.fin) if B is not None else None
    dcols = st.columns(2 if B is not None else 1)
    for col, comp, dp, color in zip(dcols, (A, B), (dpA, dpB), (C1, C2)):
        if comp is None:
            continue
        with col:
            st.markdown(f"<div class='coname' style='color:{color}'>● {esc(comp.name)}</div>", unsafe_allow_html=True)
            r = dp.iloc[-1]
            f = lambda v, p=False: "n/a" if pd.isna(v) else (f"{v:.1%}" if p else f"{v:.2f}x")
            m_ = st.columns(4)
            m_[0].metric("ROE", f(r["ROE"], True)); m_[1].metric("= Net margin", f(r["Net margin"], True))
            m_[2].metric("× Asset turnover", f(r["Asset turnover"])); m_[3].metric("× Equity multiplier", f(r["Equity multiplier"]))
            m2 = st.columns(4)
            m2[0].metric("Net margin", f(r["Net margin"], True)); m2[1].metric("= EBIT margin", f(r["EBIT margin"], True))
            m2[2].metric("× Interest burden", f(r["Interest burden"])); m2[3].metric("× Tax burden", f(r["Tax burden"]))
            with st.expander("DuPont by year"):
                show = dp.copy()
                show.index = [f"FY{y}" for y in show.index]
                st.dataframe(show.T.style.format("{:.3f}"), width="stretch")
    st.caption("Interest burden = pre-tax profit ÷ EBIT (lower means interest eats more of operating profit); tax burden = net profit ÷ pre-tax profit.")
    cr, dr_ = vA.get("cce", np.nan), vA.get("debt_ratio", np.nan)
    if not pd.isna(cr) and not pd.isna(dr_):
        cash_rich, asset_rich = cr >= 0.6, dr_ <= 0.5
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

    # ---------- how the forensic scores are calculated
    st.markdown(f"##### How the forensic scores are calculated {ADV}", unsafe_allow_html=True)
    st.caption("Every number below is traced back to the statement lines it comes from, so each score can be checked by hand.")
    fc1, fc2 = st.columns([2, 3])
    with fc1:
        fcomp = A if B is None else {A.name: A, B.name: B}[st.radio("Company", [A.name, B.name], horizontal=True, key="fx-co")]
    f_years = list(fcomp.fin.index[1:])
    with fc2:
        if len(f_years) > 1:
            fy = st.select_slider("Fiscal year", options=f_years, value=f_years[-1], format_func=lambda y: f"FY{y}", key="fx-yr")
        else:
            fy = f_years[0] if f_years else None
    fm = all_m[fcomp.ticker]
    fbf = bfA if fcomp is A else bfB
    if fy is None:
        st.info("These breakdowns need at least two years of statements.")
    else:
        ft = st.tabs(["Forensic summary", "Beneish M-score", "Piotroski F-score", "Ohlson O-score"])
        RES_TXT = {True: "✕ Flagged", False: "✓ Clear"}

        with ft[0]:
            fs = mt.forensic_summary(fm[fy], fbf)
            n, tot = fs["flagged"], fs["tested"]
            col = RED if n >= 2 else (AMBER if n == 1 or fs["watch"] else GREEN)
            head = (f"{n} of {tot} forensic tests point to aggressive accounting" if tot else "No forensic test could be run")
            meaning = ("Several independent tests agree: treat reported earnings with caution and verify them before relying on them."
                       if n >= 2 else "One isolated signal: look into that test specifically; on its own it is not conclusive."
                       if n == 1 else "No test raises a concern on the available data.")
            st.markdown(f"<div class='banner' style='--c:{col}'><div class='sig' style='font-size:1.15rem'>{esc(head)}</div>"
                        f"<div class='small'>{esc(fcomp.name)} · FY{fy} · {esc(meaning)}</div></div>", unsafe_allow_html=True)
            rows = []
            for r in fs["rows"]:
                if r["key"] == "benford":
                    val = "n/a" if r["value"] is None or pd.isna(r["value"]) else f"KS {r['value']:.3f} (cut-off {fbf['cutoff']:.3f})"
                else:
                    val = fmtv(r["key"], r["value"])
                res = "– n/a" if r["flagged"] is None else ("! Watch" if r["watch"] else RES_TXT[r["flagged"]])
                rows.append({"Test": r["test"], "Value": val, "Rule": r["rule"], "Result": res})
            fdf = pd.DataFrame(rows)
            st.dataframe(style_cells(fdf, lambda v: {"✕ Flagged": SIG_CSS["✕ Concern"], "✓ Clear": SIG_CSS["✓ Healthy"],
                                                     "! Watch": SIG_CSS["! Watch"]}.get(v, ""), ["Result"]),
                         hide_index=True, width="stretch")
            st.caption("Beneish, Sloan and the accrual ratios all lean on the gap between profit and operating cash, so they are not fully "
                       "independent. Benford uses every reported figure across all years. None of these proves fraud; they show where to dig.")

        with ft[1]:
            bd = mt.beneish_detail(fcomp.fin, fy)
            k1, k2, k3 = st.columns(3)
            k1.metric("M-score", "n/a" if pd.isna(bd["m"]) else f"{bd['m']:.2f}")
            k2.metric("Cut-off", f"{bd['cutoff']:.2f}", help="Above −2.22 the model classes the company as a likely manipulator.")
            k3.metric("Verdict", "n/a" if pd.isna(bd["m"]) else ("Likely manipulator" if bd["m"] > bd["cutoff"] else "Unlikely manipulator"))
            st.caption(f"M = −4.84 + Σ (weight × index). Each index compares FY{fy} with FY{bd['prev_year']}; 1.0 means no change.")
            fr = lambda k, v: "n/a" if v is None or pd.isna(v) else (f"Rs {v:,.0f} cr" if k == "SGI" else f"{v:.3f}")
            brows = [{"Index": f"{r['key']} · {r['name']}", f"FY{fy}": fr(r["key"], r["cur"]),
                      f"FY{bd['prev_year']}": "–" if r["prev"] is None else fr(r["key"], r["prev"]),
                      "Index value": "n/a" if pd.isna(r["index"]) else f"{r['index']:.3f}" + (" (neutral 1.0 used)" if r["imputed"] else ""),
                      "Typical: honest": r["ok_mean"], "Typical: manipulator": r["bad_mean"], "Weight": r["coef"],
                      "Adds to M": r["contribution"],
                      "Signal": "⚠ Like manipulators" if r["suspicious"] else "–"} for r in bd["rows"]]
            bdf = pd.DataFrame(brows)
            st.dataframe(bdf.style.format({"Typical: honest": "{:.3f}", "Typical: manipulator": "{:.3f}", "Weight": "{:+.3f}",
                                           "Adds to M": "{:+.3f}"}, na_rep="n/a")
                         .map(lambda v: SIG_CSS["✕ Concern"] if v == "⚠ Like manipulators" else "", subset=["Signal"]),
                         hide_index=True, width="stretch")
            if not pd.isna(bd["m"]):
                wf = go.Figure(go.Waterfall(
                    x=["Constant"] + [r["key"] for r in bd["rows"]] + ["M-score"],
                    measure=["absolute"] + ["relative"] * len(bd["rows"]) + ["total"],
                    y=[bd["const"]] + [r["contribution"] for r in bd["rows"]] + [0],
                    text=[f"{bd['const']:.2f}"] + [f"{r['contribution']:+.2f}" for r in bd["rows"]] + [f"{bd['m']:.2f}"],
                    textposition="outside", increasing=dict(marker_color=RED), decreasing=dict(marker_color=GREEN),
                    totals=dict(marker_color=RED if bd["m"] > bd["cutoff"] else GREEN), connector=dict(line=dict(color=CB))))
                wf.add_hline(y=bd["cutoff"], line_dash="dash", line_color=AMBER, annotation_text="−2.22 cut-off",
                             annotation_position="top left")
                wf.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                                 yaxis_title="Running M-score", showlegend=False)
                st.plotly_chart(wf, width="stretch", key="beneish-wf")
                st.caption("Red bars push the score towards 'manipulator', green bars pull it back. The final bar is the M-score.")
            with st.expander("What each index is looking for"):
                for r in bd["rows"]:
                    st.markdown(f"**{r['key']} - {r['name']}** `{r['formula']}`  \n{r['meaning']}"
                                + (" **This company looks like the manipulator sample on this index.**" if r["suspicious"] else ""))
                st.caption("Typical values are the sample means in Beneish (1999). An index is marked when it is closer to the manipulator "
                           "mean than to the honest mean, for indices whose weight is positive.")

        with ft[2]:
            pdt = mt.piotroski_detail(fcomp.fin, fy)
            k1, k2 = st.columns(2)
            k1.metric("F-score", "n/a" if pd.isna(pdt["score"]) else f"{pdt['score']:.0f} / 9",
                      help="7 or more is strong, 3 or less is weak.")
            k2.metric("Tests with data", f"{pdt['tested']} of 9")
            if pdt["tested"] < 9:
                st.caption("Tests without data are left out, so the score is out of fewer than 9 points.")
            prow = [{"Area": r["area"], "Test": r["test"],
                     "Result": "– no data" if r["passed"] is None else ("✓ Pass (+1)" if r["passed"] else "✕ Fail (0)"),
                     "Numbers used": r["evidence"]} for r in pdt["rows"]]
            pdf_ = pd.DataFrame(prow)
            st.dataframe(style_cells(pdf_, lambda v: SIG_CSS["✓ Healthy"] if str(v).startswith("✓") else
                                     (SIG_CSS["✕ Concern"] if str(v).startswith("✕") else ""), ["Result"]),
                         hide_index=True, width="stretch")
            st.caption(f"Comparisons are FY{fy} vs FY{pdt['prev_year']}. ROA and asset turnover use average total assets.")

        with ft[3]:
            od = mt.ohlson_detail(fcomp.fin, fy, gnp)
            if od is None:
                st.info("Ohlson inputs are missing for this year.")
            else:
                k1, k2, k3 = st.columns(3)
                k1.metric("T (log-odds)", f"{od['T']:.2f}")
                k2.metric("Probability of distress", f"{od['p']:.1%}", help="e^T / (1 + e^T). Above 50% signals default within two years.")
                k3.metric("Zone", {"G": "Low", "A": "Moderate", "R": "High"}.get(mt.rag("ohlson_p", od["p"]), "n/a"))
                orows = [{"Term": r["term"], "Input value": r["value"], "Weight": r["coef"], "Adds to T": r["contribution"]}
                         for r in od["rows"]]
                st.dataframe(pd.DataFrame(orows).style.format({"Input value": "{:.3f}", "Weight": "{:+.3f}", "Adds to T": "{:+.3f}"}),
                             hide_index=True, width="stretch")
                srt = sorted(od["rows"], key=lambda r: r["contribution"])
                of = go.Figure(go.Bar(x=[r["contribution"] for r in srt], y=[r["term"] for r in srt], orientation="h",
                                      marker_color=[RED if r["contribution"] > 0 else GREEN for r in srt]))
                of.update_layout(height=340, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                                 xaxis_title="Contribution to T (right = more risk)")
                st.plotly_chart(of, width="stretch", key="ohlson-bars")
                st.markdown(f"T = −1.32 + Σ contributions = **{od['T']:.2f}** → P = e^T ÷ (1 + e^T) = **{od['p']:.1%}**")
                st.caption(f"The size term divides total assets (Rs crore) by the price index set in the sidebar ({gnp:g}), so large Indian "
                           "companies get a large negative size term."
                           + (" This is the first year of data, so the change-in-income term uses the same year (0)." if od["first_year"] else ""))

    st.markdown("##### What drives the Altman Z″ score")
    st.caption("Weighted contribution of each term, latest year. Above 2.60 is safe, below 1.10 is distress.")
    fig = go.Figure()
    for comp, color in ((A, C1), (B, C2)):
        if comp is None:
            continue
        try:
            comps = mt.altman_components(comp.fin)
            fig.add_bar(x=[c[0] for c in comps], y=[c[1] for c in comps], name=comp.name, marker_color=color)
        except Exception:
            pass
    if fig.data:
        fig.update_layout(barmode="group", height=320, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.3), yaxis_title="Contribution to Z″")
        st.plotly_chart(fig, width="stretch", key="altman-comp")
    else:
        st.info("Altman components are not available (Yahoo is missing inputs).")
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
                     + ("**above cut-off: unusual pattern, investigate**" if r["flag"] else "within cut-off"))
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

# ------------------------------------------------------------------ ADVANCED: early warning
with tabs[5]:
    st.markdown(f"### Early warning {ADV}", unsafe_allow_html=True)
    st.caption("What has changed since the last annual report: quarterly results, the half-year balance sheet, the share price, "
               "and the market-implied Merton distance-to-default. These move months before annual ratios do.")
    if not EW_LATEST:
        st.info("Early-warning signals always use the latest quarterly and market data. Switch *As-of fiscal year* back to the "
                "latest year in View options to see them.")
    else:
        e1, e2 = st.columns([3, 1])
        with e1:
            ec = A if B is None else {A.name: A, B.name: B}[st.radio("Company", [A.name, B.name], horizontal=True, key="ew-co")]
        with e2:
            rf = st.number_input("Risk-free rate (%)", 0.0, 20.0, 6.5, 0.25, key="ew-rf",
                                 help="Used by the Merton model; roughly the 1-year Indian T-bill yield.") / 100
        with st.spinner("Loading quarterly results and share prices…"):
            ewb = ew_bundle(ec, rf)
        px_all, q_ec, mkt_all, mk_ec = ewb["px_all"], ewb["q"], ewb["mkt_all"], ewb["mk"]
        D_ec, D_src, mert, hist, sigs, n_, lvl = ewb["D"], ewb["D_src"], ewb["mert"], ewb["hist"], ewb["sigs"], ewb["n"], ewb["level"]
        st.markdown(f"<div class='banner' style='--c:{SIGNAL_COL[lvl]}'><div class='who' style='color:{C1 if ec is A else C2}'>● {esc(ec.name)}</div>"
                    f"<div class='sig'>{lvl}</div><div>{n_['R']} concern · {n_['A']} watch · {n_['G']} clear, across "
                    f"{len(sigs)} early-warning signals</div></div>", unsafe_allow_html=True)
        if not sigs:
            st.warning("Yahoo returned no quarterly or market data for this company.")
        else:
            EW_TXT = {"R": "✕ Concern", "A": "! Watch", "G": "✓ Clear", None: "–"}
            sdf = pd.DataFrame([{"Area": s_["area"], "Signal": s_["signal"], "Value": s_["value"], "Status": EW_TXT[s_["status"]],
                                 "Rule": s_["rule"], "Why it matters": s_["why"]} for s_ in sigs])
            st.dataframe(style_cells(sdf, lambda v: {"✕ Concern": SIG_CSS["✕ Concern"], "! Watch": SIG_CSS["! Watch"],
                                                     "✓ Clear": SIG_CSS["✓ Healthy"]}.get(v, ""), ["Status"]),
                         hide_index=True, width="stretch",
                         column_config={"Rule": st.column_config.TextColumn(width="medium"),
                                        "Why it matters": st.column_config.TextColumn(width="large")})

        et = st.tabs(["Quarterly results", "Merton distance-to-default", "Share price vs sector", "Sector ranking"])
        with et[0]:
            qt = ew.quarterly_table(q_ec.get("is"))
            if qt.empty:
                st.info("Yahoo has no quarterly results for this company.")
            else:
                xq = [f"{d:%b-%y}" for d in qt.index]
                qf = go.Figure()
                qf.add_bar(x=xq, y=qt["Revenue"], name="Revenue (Rs cr)", marker_color=C1 if ec is A else C2,
                           text=["" if pd.isna(v) else f"{v:+.0%} YoY" for v in qt["Revenue YoY"]], textposition="outside")
                qf.add_scatter(x=xq, y=qt["EBITDA margin"], name="EBITDA margin", yaxis="y2", mode="lines+markers",
                               line=dict(color=AMBER, width=2.5))
                qf.update_layout(height=340, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                                 yaxis=dict(title="Revenue (Rs cr)"), legend=dict(orientation="h", y=-0.2),
                                 yaxis2=dict(title="EBITDA margin", overlaying="y", side="right", tickformat=".0%", showgrid=False))
                st.plotly_chart(qf, width="stretch", key="ew-q")
                show_q = qt.copy()
                show_q.index = xq
                st.dataframe(show_q.T.style.format(lambda v: "n/a" if pd.isna(v) else f"{v:,.2f}"), width="stretch")
                st.caption("Year-on-year compares each quarter with the same quarter a year earlier, so seasonality cancels out. "
                           "Yahoo gives about five to seven quarters; gaps mean Yahoo did not publish that quarter.")

        with et[1]:
            if mert is None:
                st.info("Not enough market data (market cap or share-price history) for the Merton model.")
            elif mert["no_debt"]:
                st.success(f"{ec.name} has no debt, so there is no default point: distance-to-default is effectively infinite.")
            else:
                m1 = st.columns(5)
                m1[0].metric("Equity value (market cap)", f"Rs {money(mert['E'])} cr")
                m1[1].metric("Equity volatility", f"{mert['sigma_E']:.0%}", help="Annualised from one year of daily share-price moves.")
                m1[2].metric("Default point", f"Rs {money(mert['D'])} cr", help=f"Short-term debt + half of long-term debt, from the {D_src}.")
                m1[3].metric("Distance-to-default", f"{mert['DD']:.2f}", help="Standard deviations of asset value between today and the default point.")
                m1[4].metric("Probability of default (1y)", fpd(mert["PD"]), help="N(−DD)")
                st.markdown(f"Solved: asset value **Rs {money(mert['V'])} cr**, asset volatility **{mert['sigma_V']:.1%}**. "
                            f"DD = [ln(V ÷ D) + (r − ½σ²ᵥ)] ÷ σᵥ = [ln({mert['V']:,.0f} ÷ {mert['D']:,.0f}) + ({rf:.3f} − ½ × {mert['sigma_V']:.3f}²)] "
                            f"÷ {mert['sigma_V']:.3f} = **{mert['DD']:.2f}**")
                if len(hist):
                    hf = go.Figure()
                    top = max(float(hist["DD"].max()) * 1.1, 4)
                    hf.add_hrect(y0=0, y1=1.645, fillcolor=RED, opacity=0.10, line_width=0)
                    hf.add_hrect(y0=1.645, y1=2.326, fillcolor=AMBER, opacity=0.10, line_width=0)
                    hf.add_scatter(x=hist.index, y=hist["DD"], mode="lines+markers", line=dict(color=C1 if ec is A else C2, width=2.6),
                                   customdata=[fpd(v) for v in hist["PD"]], hovertemplate="%{x|%b-%y}: DD %{y:.2f}, PD %{customdata}<extra></extra>")
                    hf.update_layout(height=300, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                                     yaxis=dict(title="Distance-to-default", range=[0, top]))
                    st.plotly_chart(hf, width="stretch", key="ew-dd")
                    st.caption("Monthly over the last year (debt held at today's level). Red zone: DD below 1.645 = PD above 5%; "
                               "amber: DD below 2.326 = PD above 1%.")
                with st.expander("How the Merton model works"):
                    st.markdown("""
Equity holders own the firm's assets but must repay debt **D** at the horizon, so equity is a **call option** on the assets
**V**: E = V·N(d₁) − D·e^(−rT)·N(d₂). Together with σₑ·E = N(d₁)·σᵥ·V, the two observable numbers (market cap E and its volatility σₑ)
pin down the two unobservable ones (V and σᵥ). **Distance-to-default** is how many standard deviations V sits above the default
point; the model's probability of default is N(−DD). Following KMV, the default point is short-term debt plus half of long-term
debt. The model reacts to share-price moves within days, which is why it is an early-warning tool - but it is risk-neutral and
assumes log-normal assets, so read the level loosely and the **trend** carefully.""")

        with et[2]:
            if px_all.empty or ec.ticker not in px_all.columns:
                st.info("No share-price history from Yahoo.")
            else:
                rb = px_all.ffill().iloc[-ew.TRADING_DAYS * 2:]
                rb = rb / rb.bfill().iloc[0]
                pf = go.Figure()
                lo_, hi_ = rb.quantile(0.25, axis=1), rb.quantile(0.75, axis=1)
                pf.add_scatter(x=rb.index, y=hi_, line=dict(width=0), showlegend=False, hoverinfo="skip")
                pf.add_scatter(x=rb.index, y=lo_, fill="tonexty", fillcolor="rgba(139,148,158,.18)", line=dict(width=0),
                               name="Sector middle 50%", hoverinfo="skip")
                pf.add_scatter(x=rb.index, y=rb.median(axis=1), name="Sector median", line=dict(color=CB, dash="dash"))
                for comp_, col_ in ((A, C1), (B, C2)):
                    if comp_ is not None and comp_.ticker in rb:
                        pf.add_scatter(x=rb.index, y=rb[comp_.ticker], name=comp_.name, line=dict(color=col_, width=2.6))
                pf.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                                 yaxis=dict(title="Growth of Rs 1 (2 years)", tickformat=".2f"), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(pf, width="stretch", key="ew-px")
                if mk_ec:
                    mm = st.columns(4)
                    mm[0].metric("1-year return", f"{mk_ec['ret_1y']:+.1%}")
                    mm[1].metric("From 52-week high", f"{mk_ec['off_high']:+.1%}")
                    mm[2].metric("Volatility (annual)", f"{mk_ec['vol']:.0%}")
                    mm[3].metric("Worst fall in the year", f"{mk_ec['max_dd']:.1%}")

        with et[3]:
            rows_ = []
            for t, comp_ in companies.items():
                mk_ = mkt_all.get(t)
                if not mk_:
                    continue
                d_, _ = dp_for(comp_)
                r_ = ew.merton(comp_.mcap_cr, mk_["vol"], d_, rf)
                if r_:
                    rows_.append((comp_.name, r_["DD"], r_["PD"], r_["no_debt"]))
            if len(rows_) < 3:
                st.info("Too few companies have market data for a sector comparison.")
            else:
                debt_rows = sorted([r for r in rows_ if not r[3]], key=lambda r: r[1])
                cap = 15.0
                HLc = {A.name: C1, **({B.name: C2} if B is not None else {})}
                rf_ = go.Figure(go.Bar(x=[min(r[1], cap) for r in debt_rows], y=[r[0] for r in debt_rows], orientation="h",
                                       marker_color=[HLc.get(r[0], RED if r[1] < 1.645 else AMBER if r[1] < 2.326 else "rgba(139,148,158,.5)")
                                                     for r in debt_rows],
                                       customdata=[[f"{r[1]:.2f}", fpd(r[2])] for r in debt_rows],
                                       hovertemplate="%{y}: DD %{customdata[0]}, PD %{customdata[1]}<extra></extra>"))
                rf_.add_vline(x=1.645, line_color=RED, line_dash="dot")
                rf_.add_vline(x=2.326, line_color=AMBER, line_dash="dot")
                rf_.update_layout(height=max(320, 22 * len(debt_rows)), template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis=dict(title=f"Distance-to-default (capped at {cap:g} for display)"), yaxis=dict(autorange="reversed"))
                st.plotly_chart(rf_, width="stretch", key="ew-rank")
                nd_ = [r[0] for r in rows_ if r[3]]
                st.caption("Riskiest at the top. Dotted lines: PD 5% (red) and 1% (amber). Annual-report debt for every company."
                           + (f" No debt (not shown): {', '.join(nd_)}." if nd_ else ""))

# ------------------------------------------------------------------ News & ratings
with tabs[6]:
    st.markdown(f"### News & credit ratings {ADV}", unsafe_allow_html=True)
    st.caption("Headlines from the last 12 months (Google News and Yahoo Finance) that name the company, screened for credit events, "
               "plus the latest rating-agency actions set against this dashboard's grade. Headlines only - full articles are often paywalled.")
    nc = A if B is None else {A.name: A, B.name: B}[st.radio("Company", [A.name, B.name], horizontal=True, key="nw-co")]
    with st.spinner("Fetching headlines…"):
        nres = nw.get_news(nc.name, nc.ticker)
    items = nres["items"]
    if nres["errors"]:
        st.caption("Some news sources did not respond: " + ", ".join(nres["errors"]) + ". Use *Reload data* to try again.")
    ns = nw.news_signal(items)
    NS_TXT = {"R": ("CONCERN", RED), "A": ("WATCH", AMBER), "G": ("CLEAR", GREEN), None: ("NO HEADLINES", "#8b949e")}[ns["status"]]
    st.markdown(f"<div class='banner' style='--c:{NS_TXT[1]}'><div class='who' style='color:{C1 if nc is A else C2}'>● {esc(nc.name)}</div>"
                f"<div class='sig'>{NS_TXT[0]}</div><div>News flow (keyword screen) · {len(items)} headlines in 12 months · "
                f"{ns['value']}</div></div>", unsafe_allow_html=True)
    if ns["trigger"]:
        st.caption(f"Driven by: “{ns['trigger']}”. This is the *News flow* signal in the Early warning tab.")

    # ---- ratings
    st.markdown("#### Credit ratings: agencies vs this model")

    def model_grade(t):
        """Grade from the quantitative score on textbook cut-offs (no early-warning or analyst notches)."""
        mt.set_basis("Textbook")
        s_ = mt.overall_score(mt.pillar_scores(latest_all[t], companies[t].gov), W)
        mt.set_basis(basis_sel, SECTOR_VALS)
        return rp.grade_index(s_), s_

    def gap_words(gap):
        if gap is None:
            return "–"
        return "same grade" if gap == 0 else f"model {abs(gap)} grade{'s' * (abs(gap) > 1)} {'lower' if gap > 0 else 'higher'}"

    gi, gsc = model_grade(nc.ticker)
    r1, r2 = st.columns([1, 3])
    with r1:
        if gi is None:
            st.metric("Model grade", "n/a")
        else:
            st.metric("Model grade (score only)", rp.grade_info(gi)["label"], f"score {gsc:.0f}/100", delta_color="off",
                      help="Quantitative score on textbook cut-offs mapped to CR1-CR9, before early-warning and analyst notches - "
                           "the same base grade as the Credit memo.")
        with st.popover("Enter a rating you looked up", width="stretch"):
            st.caption("From the agency's website or the company's annual report. Kept for this session only.")
            m_ag = st.selectbox("Agency", ["CRISIL", "ICRA", "CARE", "India Ratings", "Acuite", "Brickwork"], key=f"nw-man-ag-{nc.ticker}")
            m_rt = st.selectbox("Long-term rating", ["–", "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB",
                                                     "BB-", "B+", "B", "B-", "C", "D"], key=f"nw-man-rt-{nc.ticker}")
            m_ol = st.selectbox("Outlook", ["Stable", "Positive", "Negative", "Developing", "Watch"], key=f"nw-man-ol-{nc.ticker}")
    rats = nw.latest_ratings(items)
    if m_rt != "–":
        rats = [dict(agency=m_ag, rating=m_rt, outlook=m_ol, action="Entered by analyst", scale="national",
                     date=datetime.now(timezone.utc), title="Entered manually", url="", source="")] + \
               [r_ for r_ in rats if r_["agency"] != m_ag]
    with r2:
        if not rats:
            st.info("No rating action named this company in the last 12 months of headlines. Agencies publish every rating on their "
                    "own sites (CRISIL, ICRA, CARE, India Ratings) - check there before relying on the model grade alone.")
        else:
            rrows = []
            for r_ in rats:
                ai_ = nw.grade_index_of(r_["rating"]) if r_["scale"] == "national" else None
                gap = (gi - ai_) if (gi is not None and ai_ is not None) else None
                rrows.append({"Agency": r_["agency"], "Scale": r_["scale"].title(), "Rating": r_["rating"] or "–",
                              "Outlook": r_["outlook"] or "–", "Action": r_["action"] or "–", "Date": f"{r_['date']:%d %b %Y}",
                              "≈ CR grade": rp.GRADES[ai_][1] if ai_ is not None else "–", "Gap": gap_words(gap),
                              "Headline": r_["title"], "Link": r_["url"] or None})
            st.dataframe(pd.DataFrame(rrows), hide_index=True, width="stretch",
                         column_config={"Link": st.column_config.LinkColumn(display_text="open"),
                                        "Headline": st.column_config.TextColumn(width="large")})
    nat = [r_ for r_ in rats if r_["scale"] == "national" and nw.grade_index_of(r_["rating"]) is not None]
    if nat and gi is not None:
        r_ = nat[0]
        gap = gi - nw.grade_index_of(r_["rating"])
        if abs(gap) <= 1:
            st.success(f"The model broadly agrees with {r_['agency']} ({r_['rating']}): {gap_words(gap)}.")
        elif gap > 0:
            st.warning(f"The model is {gap} grades below {r_['agency']} ({r_['rating']}). Agencies also weigh what the statements do not show: "
                       "market position, parent or state support, regulated returns and access to funding. Textbook cut-offs are also harsh "
                       "on capital-heavy sectors. Read the agency's rationale before treating the gap as a red flag.")
        else:
            st.error(f"The model is {-gap} grades above {r_['agency']} ({r_['rating']}). The agency may be seeing risks the ratios miss - "
                     "contingent liabilities, group exposure, refinancing walls or governance. Read its rationale before relying on the model.")
    if rats:
        st.caption("Ratings are read from headlines, so check the entity named (group companies share names) and the date. "
                   "Moody's, Fitch and S&P use a global scale capped near India's sovereign rating (BBB- / Baa3), so they are shown but "
                   "not compared with the national-scale CR grade.")
    if not EW_LATEST:
        st.caption("The model grade above uses the as-of year chosen in View options; ratings and headlines are always current.")

    with st.expander(f"Model vs agencies across the {sector} sector"):
        st.caption("Searches rating-agency headlines for every company in the sector (one search each, cached 6 hours) and lines up the "
                   "latest national-scale rating against the model grade. A real-world check on how far the model can be trusted.")
        skey = f"nw-sector-{sector}-{asof_y}"
        if st.button("Run the sector check", key="nw-sector-btn"):
            def one(t):
                r_ = nw.get_news(companies[t].name, t, sources=("Rating agencies",))
                return t, [x for x in nw.latest_ratings(r_["items"]) if x["scale"] == "national" and nw.grade_index_of(x["rating"]) is not None]
            from concurrent.futures import ThreadPoolExecutor
            with st.spinner(f"Searching rating news for {len(companies)} companies…"), ThreadPoolExecutor(8) as ex_:
                st.session_state[skey] = dict(ex_.map(one, list(companies)))
        found = st.session_state.get(skey)
        if found is not None:
            srows = []
            for t, lst in found.items():
                g_, s_ = model_grade(t)
                if not lst or g_ is None:
                    continue
                r_ = lst[0]
                ag_i = nw.grade_index_of(r_["rating"])
                srows.append({"Company": companies[t].name, "Model": rp.GRADES[g_][1], "Score": round(s_),
                              "Agency": f"{r_['agency']} {r_['rating']}", "≈ CR grade": rp.GRADES[ag_i][1], "gap": g_ - ag_i,
                              "Gap": gap_words(g_ - ag_i), "Date": f"{r_['date']:%b %Y}", "_m": g_, "_a": ag_i})
            if not srows:
                st.info("No national-scale ratings found in headlines for this sector.")
            else:
                sdf_ = pd.DataFrame(srows).sort_values("gap", key=abs, ascending=False)
                within = (sdf_["gap"].abs() <= 1).mean()
                k1, k2, k3 = st.columns(3)
                k1.metric("Companies with a rating found", f"{len(sdf_)} of {len(companies)}")
                k2.metric("Within one grade", f"{within:.0%}")
                k3.metric("Average gap", f"{sdf_['gap'].mean():+.1f} grades",
                          help="Positive = the model is harsher than the agencies on average.")
                jit = np.linspace(-0.18, 0.18, len(sdf_))
                fig_ = go.Figure()
                fig_.add_trace(go.Scatter(x=[0, 8], y=[0, 8], mode="lines", line=dict(color="#8b949e", dash="dot"), hoverinfo="skip",
                                          name="agreement"))
                fig_.add_trace(go.Scatter(x=sdf_["_a"] + jit, y=sdf_["_m"] - jit, mode="markers", name="company",
                                          marker=dict(size=11, color=[C1 if n_ == A.name else (C2 if B is not None and n_ == B.name else "#57606a")
                                                                      for n_ in sdf_["Company"]]),
                                          text=sdf_["Company"] + " · " + sdf_["Agency"] + " · model " + sdf_["Model"],
                                          hovertemplate="%{text}<extra></extra>"))
                ticks = dict(tickmode="array", tickvals=list(range(9)), ticktext=[f"{g[1]}<br>≈{g[2]}" for g in rp.GRADES])
                fig_.update_layout(height=420, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                                   xaxis=dict(title="Agency rating (mapped)", range=[-0.5, 8.5], **ticks),
                                   yaxis=dict(title="Model grade", range=[8.5, -0.5], **ticks))
                st.plotly_chart(fig_, width="stretch", key="nw-sector-fig")
                st.caption("Dots on the dotted line agree; below it the model is harsher than the agency, above it more lenient.")
                st.dataframe(sdf_.drop(columns=["gap", "_m", "_a"]), hide_index=True, width="stretch")

    # ---- AI
    st.markdown("#### AI read of the headlines")
    ai_key = f"nw-ai-{nc.ticker}"
    sent = nw.ai_input(items)
    if not nw.api_key():
        st.info("Optional: add an Anthropic API key to have Claude read these headlines, judge each one for lenders, pick out rating "
                "actions and write a short digest with follow-up questions.  \n"
                "**Streamlit Cloud:** app menu → Settings → Secrets → `ANTHROPIC_API_KEY = \"sk-ant-...\"`.  \n"
                "**Locally:** `.streamlit/secrets.toml` with the same line (that file is git-ignored). "
                "Everything else on this tab works without it.")
    elif not sent:
        st.caption("No credit-relevant headlines to analyse.")
    else:
        a1, a2 = st.columns([1, 3])
        if a1.button(f"Analyse {len(sent)} headlines with AI", key=f"nw-ai-btn-{nc.ticker}", type="primary"):
            with st.spinner(f"{nw.AI_MODEL} is reading the headlines…"):
                try:
                    st.session_state[ai_key] = (nw.ai_analyse(nc.name, sector, sent), [it["title"] for it in sent])
                except RuntimeError as e:
                    st.error(str(e))
        a2.caption("One request per click, cached 6 hours. Share-price chatter is left out. The AI sees only headlines and short snippets, "
                   "can misread them, and does not change the grade or the early-warning signal - treat it as a reading aid.")
    ai_res = st.session_state.get(ai_key)
    ai_map = {}
    if ai_res:
        ai, sent_titles = ai_res
        sent_items = [next((it for it in items if it["title"] == t_), {"title": t_, "url": ""}) for t_ in sent_titles]
        ai_map = nw.ai_by_title(sent_items, ai)
        RISK_COL = {"Low": GREEN, "Moderate": AMBER, "High": RED}
        st.markdown(f"<div class='banner' style='--c:{RISK_COL.get(ai['news_risk'], '#8b949e')}'><div class='who'>AI view</div>"
                    f"<div class='sig'>{ai['news_risk'].upper()} NEWS RISK</div><div>{len(ai_map)} headlines read</div></div>",
                    unsafe_allow_html=True)
        st.markdown(ai["digest"])
        if ai.get("watch_points"):
            st.markdown("**Questions for management**  \n" + "\n".join(f"- {w}" for w in ai["watch_points"]))
        if ai.get("ratings"):
            st.markdown("**Rating actions the AI found**")
            st.dataframe(pd.DataFrame([{"Agency": r_["agency"], "Rating": r_["rating"] or "–", "Outlook": r_["outlook"] or "–",
                                        "Action": r_["action"] or "–",
                                        "Headline": sent_items[r_["id"]]["title"] if 0 <= r_["id"] < len(sent_items) else ""}
                                       for r_ in ai["ratings"]]), hide_index=True, width="stretch")
        dis = sum(1 for it in items if it["title"] in ai_map and ai_map[it["title"]]["relevant"] and ai_map[it["title"]]["tone"] != it["tone"])
        irr = sum(1 for x in ai_map.values() if not x["relevant"])
        with st.expander("Headlines the digest cites"):
            for i_, it in enumerate(sent_items):
                st.markdown(f"[{i_}] [{esc(it['title'])}]({it.get('url', '')})")
        st.caption(f"The AI disagrees with the keyword screen on the tone of {dis} headline(s) and judged {irr} not really about "
                   f"{nc.name}. Model {ai.get('model')}, {ai['usage']['in']:,} input / {ai['usage']['out']:,} output tokens.")

    # ---- headlines
    st.markdown("#### Headlines")
    if not items:
        st.info("No headlines naming this company in the last 12 months.")
    else:
        f1, f2, f3 = st.columns([1, 2, 2])
        hide_mkt = f1.toggle("Hide share-price chatter", True, key="nw-hide-mkt")
        tones = f2.segmented_control("Tone", ["negative", "neutral", "positive"], selection_mode="multi",
                                     default=["negative", "neutral", "positive"], key="nw-tone") or []
        cats = f3.multiselect("Category", nw.CATEGORIES, key="nw-cat", placeholder="All categories")
        view = [it for it in items if (not hide_mkt or it["category"] != "Market commentary") and it["tone"] in tones
                and (not cats or it["category"] in cats)]
        TONE_COL = {"negative": RED, "neutral": "#8b949e", "positive": GREEN}
        mdf = pd.DataFrame([{"m": it["date"].strftime("%Y-%m"), "tone": it["tone"]} for it in view])
        if not mdf.empty:
            fig_ = go.Figure()
            for tn in ("negative", "neutral", "positive"):
                cnt = mdf[mdf["tone"] == tn].groupby("m").size()
                fig_.add_trace(go.Bar(x=cnt.index, y=cnt.values, name=tn, marker_color=TONE_COL[tn]))
            fig_.update_layout(barmode="stack", height=230, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=-0.3), xaxis=dict(type="category"), yaxis_title="headlines")
            st.plotly_chart(fig_, width="stretch", key="nw-timeline")
        TONE_TXT = {"negative": "✕ negative", "neutral": "· neutral", "positive": "✓ positive"}
        hrows = []
        for it in view:
            row = {"Date": it["date"].strftime("%d %b %Y"), "Headline": it["title"], "Source": it["source"] or it["via"],
                   "Category": it["category"], "Tone": TONE_TXT[it["tone"]], "Severity": it["severity"]}
            x = ai_map.get(it["title"])
            if ai_map:
                row["AI tone"] = ("not about company" if not x["relevant"] else TONE_TXT[x["tone"]]) if x else "–"
                row["AI note"] = x["note"] if x else ""
            row["Link"] = it["url"]
            hrows.append(row)
        hdf = pd.DataFrame(hrows)
        tcss = lambda v: {"✕ negative": SIG_CSS["✕ Concern"], "✓ positive": SIG_CSS["✓ Healthy"]}.get(v, "")
        st.dataframe(style_cells(hdf, tcss, [c for c in ("Tone", "AI tone") if c in hdf.columns]) if len(hdf) else hdf,
                     hide_index=True, width="stretch", height=min(600, 38 + 35 * len(hdf)),
                     column_config={"Link": st.column_config.LinkColumn(display_text="open"),
                                    "Headline": st.column_config.TextColumn(width="large"),
                                    "AI note": st.column_config.TextColumn(width="large")})
        st.caption(f"{len(view)} of {len(items)} headlines shown. Tone and severity come from a transparent keyword screen "
                   "(see Data & method); it can misread unusual wording.")


# ------------------------------------------------------------------ Cash flow & lifecycle
with tabs[7]:
    st.subheader("Cash flow & lifecycle")
    st.caption("Where the cash came from and where it went, and what stage of life the business is in.")
    yrs = sorted(set(A.fin.index) | (set(B.fin.index) if B is not None else set()))
    yr = st.select_slider("Fiscal year", options=yrs, value=yrs[-1], format_func=lambda y: f"FY{y}")
    wcols = st.columns(2 if B is not None else 1)
    for col, comp, color in zip(wcols, (A, B), (C1, C2)):
        if comp is None:
            continue
        with col:
            y_ = yr if yr in comp.fin.index else comp.fin.index[-1]
            steps, net, approx = mt.waterfall_steps(comp.fin.loc[y_])
            fig = go.Figure(go.Waterfall(
                x=[s[0] for s in steps] + ["Net change in cash"], y=[s[1] for s in steps] + [0],
                measure=["relative"] * len(steps) + ["total"], connector=dict(line=dict(color=CB, width=1)),
                increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
                totals=dict(marker=dict(color=color)), text=[f"{s[1]:,.0f}" for s in steps] + [f"{net:,.0f}"], textposition="outside"))
            fig.update_layout(title=dict(text=f"{comp.name} - FY{y_} (Rs cr)", font=dict(size=14)), height=380,
                              template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig, width="stretch", key=f"wf-{comp.ticker}")
            if approx:
                st.caption("Yahoo does not split every cash-flow line for this company, so 'other' items are approximate.")
    st.markdown("##### Lifecycle stage by year")
    st.caption("Stage from the signs of operating (CFO), investing (CFI) and financing (CFF) cash flow.")
    for comp in (A, B):
        if comp is None:
            continue
        rows = []
        for y_, r in comp.fin.iterrows():
            stage, why = mt.lifecycle(r.get("cfo", np.nan), r.get("cfi", np.nan), r.get("cff", np.nan))
            rows.append({"Year": f"FY{y_}", "CFO": r.get("cfo", np.nan), "CFI": r.get("cfi", np.nan), "CFF": r.get("cff", np.nan),
                         "Stage": stage or "n/a", "Credit reading": why or "Yahoo is missing a cash-flow total"})
        st.markdown(f"**{comp.name}**")
        st.dataframe(pd.DataFrame(rows).style.format({"CFO": "{:,.0f}", "CFI": "{:,.0f}", "CFF": "{:,.0f}"}, na_rep="n/a"),
                     width="stretch", hide_index=True)

# ------------------------------------------------------------------ ADVANCED: credit workbench
with tabs[8]:
    st.markdown(f"### Credit workbench {ADV}", unsafe_allow_html=True)
    st.caption("Decision tools built on top of the core ratios. Assumptions are simplified and listed in *Data & method*.")
    bor_name = st.radio("Borrower", [A.name] + ([B.name] if B is not None else []), horizontal=True, key="borrower") if B is not None else A.name
    BC = A if bor_name == A.name else B
    bm = all_m[BC.ticker]; bvals = latest_all[BC.ticker]
    bflags = flagsA if BC is A else flagsB
    brow = BC.fin.iloc[-1]
    g_ = lambda k: brow[k] if k in brow.index else np.nan
    base = dict(rev=g_("rev"), ebitda=g_("ebitda"), ebit=g_("ebit"), da=g_("da"), interest=g_("interest"), debt=g_("debt"),
                cur_debt=g_("cur_debt"), cash=g_("cash"), recv=g_("recv"), inv=g_("inv"), pay=g_("pay"), cfo=g_("cfo"),
                eq=g_("equity"), ni=g_("ni"))
    taxr = float(np.clip(g_("tax") / g_("pbt"), 0, 0.4)) if (not pd.isna(g_("tax")) and not pd.isna(g_("pbt")) and g_("pbt") > 0) else 0.25
    wb = st.tabs(["Facility & stress", "Covenants", "Rating scorecard", "Sector watchlist", "Credit memo"])
    fac = {}

    # ---- facility & stress
    with wb[0]:
        st.markdown(f"##### Facility request and stress test {ADV}", unsafe_allow_html=True)
        implied = base["interest"] / base["debt"] if base["debt"] and base["debt"] > 0 and not pd.isna(base["interest"]) else 0.095
        k_ = BC.ticker
        c = st.columns(5)
        ftype = c[0].selectbox("Facility type", ["Term loan (amortising)", "Working-capital limit (revolving)"], key=f"ft_{k_}")
        revolving = ftype.startswith("Working")
        default_amt = float(max(round(0.25 * (base["ebitda"] if not pd.isna(base["ebitda"]) else 100), -1), 10.0))
        amount = c[1].number_input("Amount (Rs cr)", 0.0, 1e7, default_amt, 10.0, key=f"fa_{k_}")
        rate = c[2].number_input("Interest rate (%)", 0.0, 30.0, float(round(min(max(implied * 100, 7.0), 14.0), 2)), 0.25, key=f"fr_{k_}") / 100
        tenor = c[3].number_input("Tenor (years)", 1.0, 30.0, 5.0, 1.0, key=f"fn_{k_}", disabled=revolving)
        target = c[4].number_input("Target DSCR (x)", 0.5, 3.0, 1.25, 0.05, key=f"fd_{k_}")
        st.markdown("**Stress scenario**")
        s_ = st.columns(4)
        sg = s_[0].slider("Revenue change (%)", -40, 30, -10, key=f"sg_{k_}") / 100
        sdm = s_[1].slider("EBITDA margin change (pp)", -10.0, 5.0, -2.0, 0.5, key=f"sdm_{k_}") / 100
        sdr = s_[2].slider("Interest-rate change (pp)", -2.0, 5.0, 1.0, 0.25, key=f"sdr_{k_}") / 100
        sdd = s_[3].slider("Receivable days change", -20, 90, 10, key=f"sdd_{k_}")
        kw = dict(new_rate=rate, tenor=tenor, revolving=revolving, tax_rate=taxr)
        r_base = mt.stress(base, tax_rate=taxr)
        r_pro = mt.stress(base, new_debt=amount, **kw)
        r_str = mt.stress(base, g=sg, dm=sdm, dr=sdr, d_recv_days=sdd, new_debt=amount, **kw)
        capacity = mt.debt_capacity(base, target, rate, tenor, revolving)
        covs = {k: st.session_state.get(f"cov_{k}", spec[2]) for k, spec in mt.COVENANT_SPECS.items()}
        rows = [("Revenue (Rs cr)", "rev", "{:,.0f}"), ("EBITDA (Rs cr)", "ebitda", "{:,.0f}"), ("Interest (Rs cr)", "interest", "{:,.1f}"),
                ("Operating cash flow (Rs cr)", "cfo", "{:,.0f}"), ("Interest cover", "icr", "{:.2f}x"), ("DSCR", "dscr", "{:.2f}x"),
                ("Debt / Equity", "de", "{:.2f}x"), ("Net debt / EBITDA", "nd_ebitda", "{:.2f}x"),
                ("Cash conversion (CFO/EBITDA)", "cce", "{:.0%}"), ("Cash runway (months)", "runway_m", "{:.0f}")]
        tbl, css = [], []
        for label, k, f in rows:
            vals_ = [r_base[k], r_pro[k], r_str[k]]
            tbl.append([label] + [("–" if pd.isna(v) else f.format(v)) for v in vals_])
            cs = [""]
            for v in vals_:
                st_ = ""
                if k in mt.COVENANT_SPECS and not pd.isna(v):
                    status, _ = mt.covenant_status(v, covs[k], mt.COVENANT_SPECS[k][1])
                    st_ = {"Breach": "background-color: rgba(248,81,73,.32)", "Tight": "background-color: rgba(210,153,34,.32)"}.get(status, "")
                cs.append(st_)
            css.append(cs)
        tdf = pd.DataFrame(tbl, columns=["", "Current", "With facility", "Stressed (facility + shocks)"])
        cdf = pd.DataFrame(css, columns=tdf.columns)
        st.dataframe(tdf.style.apply(lambda _: cdf, axis=None), width="stretch", hide_index=True)
        st.caption("Amber = within 15% of a covenant limit, red = breach (limits are set on the Covenants tab).")
        m3 = st.columns(3)
        m3[0].metric("Largest facility at target DSCR", money(capacity))
        m3[1].metric("Requested vs capacity", "No room" if capacity == 0 and amount > 0 else ("Within capacity" if amount <= capacity else "Exceeds capacity"))
        m3[2].metric("Pro-forma DSCR → stressed", f"{r_pro['dscr']:.2f}x → {r_str['dscr']:.2f}x" if not (pd.isna(r_pro['dscr']) or pd.isna(r_str['dscr'])) else "n/a")
        st.markdown("**DSCR sensitivity: EBITDA margin vs interest rate** (with the facility and the revenue / receivable shocks above)")
        dm_list, dr_list = [-0.06, -0.04, -0.02, 0.0, 0.02], [0.0, 0.01, 0.02, 0.03, 0.04]
        grid = mt.sens_grid(base, dm_list, dr_list, "dscr", g=sg, d_recv_days=sdd, new_debt=amount, **kw)
        z = grid.values.astype(float)
        cls = np.where(np.isnan(z), np.nan, np.where(z < 1.0, 0, np.where(z < target, 1, 2)))
        hm = go.Figure(go.Heatmap(z=cls, x=[f"{d*100:+.0f}pp" for d in dm_list], y=[f"+{d*100:.0f}pp" for d in dr_list],
                                  colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]], zmin=0, zmax=2, showscale=False,
                                  text=[[("n/a" if pd.isna(v) else f"{v:.2f}x") for v in row] for row in z], texttemplate="%{text}",
                                  hovertemplate="margin %{x}, rates %{y}: %{text}<extra></extra>"))
        hm.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="Change in EBITDA margin", yaxis_title="Change in interest rate")
        st.plotly_chart(hm, width="stretch", key="sens-heat")
        st.caption(f"Red: DSCR below 1.0 · amber: below the {target:.2f}x target · green: at or above target.")
        fac.update(dict(req=dict(type=ftype, amount=amount, rate=rate, tenor=tenor, target=target, capacity=capacity),
                        scen=dict(base=r_base, pro=r_pro, stress=r_str,
                                  desc=f"revenue {sg:+.0%}, EBITDA margin {sdm*100:+.1f}pp, rates {sdr*100:+.2f}pp, receivable days {sdd:+d}"),
                        grid=grid, dm_list=dm_list, dr_list=dr_list))

    # ---- covenants
    with wb[1]:
        st.markdown(f"##### Covenant monitor {ADV}", unsafe_allow_html=True)
        st.caption("Set the limits a lender might impose; the table shows actual value and status for each year.")
        cc = st.columns(len(mt.COVENANT_SPECS))
        for col, (k, (label, kind, dflt)) in zip(cc, mt.COVENANT_SPECS.items()):
            col.number_input(label, 0.0, 50.0, float(dflt), 0.05, key=f"cov_{k}")
        covs = {k: st.session_state.get(f"cov_{k}", spec[2]) for k, spec in mt.COVENANT_SPECS.items()}
        yrs_c = list(bm.columns)[-4:]
        rows, css = [], []
        latest_rows = []
        for k, (label, kind, _) in mt.COVENANT_SPECS.items():
            r_, c_ = [f"{label.split(' (')[0]} ({'≥' if kind == 'min' else '≤'} {covs[k]:g})"], [""]
            for y in yrs_c:
                v = bm.loc[k, y] if k in bm.index else np.nan
                status, head = mt.covenant_status(v, covs[k], kind)
                r_.append("n/a" if pd.isna(v) else f"{fmtv(k, v)} · {status}")
                c_.append({"Breach": "background-color: rgba(248,81,73,.32)", "Tight": "background-color: rgba(210,153,34,.32)",
                           "OK": "background-color: rgba(46,160,67,.25)"}.get(status, ""))
            v_last = bvals.get(k)
            latest_rows.append((label.split(" (")[0], fmtv(k, v_last), f"{'≥' if kind == 'min' else '≤'} {covs[k]:g}",
                                mt.covenant_status(v_last, covs[k], kind)[0]))
            rows.append(r_); css.append(c_)
        cdf = pd.DataFrame(rows, columns=["Covenant"] + [f"FY{y}" for y in yrs_c])
        cdf_css = pd.DataFrame(css, columns=cdf.columns)
        st.dataframe(cdf.style.apply(lambda _: cdf_css, axis=None), width="stretch", hide_index=True)
        n_b = sum(1 for r in latest_rows if r[3] == "Breach"); n_t = sum(1 for r in latest_rows if r[3] == "Tight")
        k3 = st.columns(3)
        k3[0].metric("Breaches (latest year)", n_b); k3[1].metric("Tight (<15% headroom)", n_t)
        k3[2].metric("Passing", sum(1 for r in latest_rows if r[3] == "OK"))
        e_, i_ = base["ebit"], base["interest"]
        if not pd.isna(e_) and not pd.isna(i_) and e_ > 0 and i_ > 0:
            cushion = 1 - covs["icr"] * i_ / e_
            st.write(f"**EBIT cushion:** operating profit could fall by **{max(cushion, 0):.0%}** before the interest-cover covenant "
                     f"({covs['icr']:g}x) is breached.")
        fac["cov_rows"] = latest_rows

    # ---- rating scorecard
    with wb[2]:
        st.markdown(f"##### Indicative risk scorecard {ADV}", unsafe_allow_html=True)
        st.caption("Weighted blend of the factor scores. It is an internal-style indication, not an agency rating. "
                   "Change the weights to reflect your view; the banner and Overview update.")
        with st.expander("Adjust weights", expanded=False):
            wc = st.columns(6)
            for col, (p, d) in zip(wc, mt.DEFAULT_WEIGHTS.items()):
                col.slider(p, 0, 50, d, key=f"w_{p}")
        W = {p: st.session_state.get(f"w_{p}", d) for p, d in mt.DEFAULT_WEIGHTS.items()}
        pb = mt.pillar_scores(bvals, BC.gov)
        pcmp = pB if BC is A else pA
        cmp_nm = cmp_name if BC is A else A.name
        sc_b, sc_c = mt.overall_score(pb, W), mt.overall_score(pcmp, W)
        k2 = st.columns(3)
        k2[0].metric(f"{BC.name} overall", "n/a" if pd.isna(sc_b) else f"{sc_b:.0f} / 100")
        k2[1].metric("Indicative band", mt.band(sc_b))
        k2[2].metric(f"{cmp_nm}", "n/a" if pd.isna(sc_c) else f"{sc_c:.0f} / 100 · {mt.band(sc_c)}")
        fig = go.Figure()
        fig.add_bar(y=list(pb), x=[0 if pd.isna(v) else v for v in pb.values()], orientation="h", name=BC.name,
                    marker_color=C1 if BC is A else C2)
        fig.add_bar(y=list(pcmp), x=[0 if pd.isna(v) else v for v in pcmp.values()], orientation="h", name=cmp_nm, marker_color=CB)
        fig.update_layout(barmode="group", height=380, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(range=[0, 100], title="Factor score (0-100)"), yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", key="pillar-bars")
        st.caption("Bands: 80+ low risk · 65-80 moderate · 50-65 elevated · 35-50 high · below 35 very high. "
                   "Governance uses Yahoo/ISS risk scores and is left out (weights re-normalised) when unavailable.")
        fac["score"] = (sc_b, mt.band(sc_b), pb)

    # ---- sector watchlist
    with wb[3]:
        st.markdown(f"##### Sector watchlist - {sector} {ADV}", unsafe_allow_html=True)
        st.caption("Every loaded company ranked riskiest first by indicative score. Δ is the change in the score versus the prior fiscal year.")
        rows = []
        for t, c in companies.items():
            m_ = all_m[t]
            ys = mt.score_by_year(m_, W)
            now, prev = ys.iloc[-1], (ys.iloc[-2] if len(ys) > 1 else np.nan)
            v_ = latest_all[t]
            vd = ins.verdict(v_, mt.red_flags(c.fin, m_))
            d_ = now - prev if not pd.isna(now) and not pd.isna(prev) else np.nan
            rows.append({"": "◀" if t in (t1, t2) else "", "Company": c.name, "Score": now, "Band": mt.band(now),
                         "Δ vs prior yr": ("n/a" if pd.isna(d_) else f"{'▲' if d_ > 0 else '▼' if d_ < 0 else '■'} {abs(d_):.0f}"),
                         "Signal": vd["signal"].title(), "Concerns": vd["r"],
                         "Interest cover": fmtv("icr", v_.get("icr")), "DSCR": fmtv("dscr", v_.get("dscr")),
                         "Net debt/EBITDA": fmtv("nd_ebitda", v_.get("nd_ebitda")), "Cash conv.": fmtv("cce", v_.get("cce")),
                         "Altman Z″": fmtv("altman_z2", v_.get("altman_z2")), "Ohlson P(distress)": fmtv("ohlson_p", v_.get("ohlson_p"))})
        wdf = pd.DataFrame(rows).sort_values("Score", na_position="first")
        st.dataframe(wdf, column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")},
                     width="stretch", hide_index=True)
        st.download_button("Download watchlist (CSV)", wdf.to_csv(index=False), file_name=f"watchlist_{sector}.csv", mime="text/csv", key="dl-watch")

    # ---- credit memo
    with wb[4]:
        st.markdown(f"##### Credit appraisal note {ADV}", unsafe_allow_html=True)
        st.caption("Assembled from every other tab (facility, stress, covenants, scorecard, early warning) plus your own judgement "
                   "below. Download it as PDF or Word with the charts included. A draft for analyst review, not a credit decision.")
        k_ = BC.ticker
        req_amt = (fac.get("req") or {}).get("amount", 0.0) or 0.0
        with st.container(border=True):
            st.markdown("**Analyst inputs**")
            a1, a2, a3 = st.columns(3)
            an_name = a1.text_input("Analyst", key=f"m_name_{k_}", placeholder="Your name")
            purpose = a2.text_input("Purpose of the facility", key=f"m_purpose_{k_}", placeholder="e.g. capex for a new plant")
            mgmt = a3.selectbox("Management quality", ["Adequate", "Strong", "Weak"], key=f"m_mgmt_{k_}",
                                help="Strong +1 notch, Weak −1 notch on the issuer grade.")
            b1, b2, b3 = st.columns(3)
            ctype = b1.selectbox("Security", ["Unsecured", "Fixed assets (property, plant)", "Receivables & inventory",
                                              "Financial securities / deposits", "Corporate or promoter guarantee"], key=f"m_ctype_{k_}")
            cval = b2.number_input("Security value (Rs cr)", 0.0, 1e8, 0.0, 10.0, key=f"m_cval_{k_}", disabled=ctype == "Unsecured",
                                   help="Realisable value after haircut. Cover = value ÷ facility amount; 1.5x or more lifts the facility grade a notch.")
            outlook = b3.selectbox("Industry outlook", ["Stable", "Positive", "Negative"], key=f"m_outlook_{k_}",
                                   help="Negative −1 notch on the issuer grade.")
            notes = st.text_area("Analyst notes (go into the memo)", key=f"m_notes_{k_}", height=80,
                                 placeholder="Business model, customer concentration, promoter track record, site visit…")
            o1, o2 = st.columns([1, 2])
            override = o1.selectbox("Recommendation", ["Use system stance", "Supportable", "Supportable with conditions",
                                                       "Not supportable on current numbers"], key=f"m_ovr_{k_}")
            ov_reason = o2.text_input("Reason for overriding the system stance", key=f"m_ovr_why_{k_}",
                                      disabled=override == "Use system stance",
                                      placeholder="Required when overriding - it is printed in the memo")
        if override != "Use system stance" and not ov_reason.strip():
            st.warning("Add a reason for the override - the memo will use the system stance until you do.")
            override = "Use system stance"
        cover = (cval / req_amt) if (ctype != "Unsecured" and req_amt > 0) else np.nan

        sc_b, band_b, pb_ = fac.get("score", (np.nan, "n/a", {}))
        # A grade is an absolute statement, so it always comes from textbook-basis inputs (score and early-warning statuses),
        # whatever basis the screen shows; the sector-relative score is reported alongside as relative standing.
        mt.set_basis("Textbook")
        tb_score = mt.overall_score(mt.pillar_scores(bvals, BC.gov), W)
        ew_b = ew_bundle(BC, st.session_state.get("ew-rf", 6.5) / 100) if EW_LATEST else None
        ew_ctx = dict(level=ew_b["level"], n=ew_b["n"], sigs=ew_b["sigs"], mert=ew_b["mert"]) if ew_b and ew_b["sigs"] else None
        mt.set_basis("Sector-relative", SECTOR_VALS)
        sr_score = mt.overall_score(mt.pillar_scores(bvals, BC.gov), W)
        mt.set_basis(basis_sel, SECTOR_VALS)                    # restore the session's basis
        rating = rp.rate(tb_score, ew_ctx["level"] if ew_ctx else None, mgmt, outlook, cover)
        if rating:
            rating["relative"] = rp.relative_standing(sr_score, tb_score, sector)

        if rating:
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Issuer grade", rating["issuer"]["label"], help=rating["issuer"]["meaning"])
            g2.metric("Facility grade", rating["facility"]["label"], help="Issuer grade, lifted one notch by security cover of 1.5x or more.")
            g3.metric("Indicative 1-year PD", f"{rating['facility']['pd']:.2%}",
                      help="Rounded long-run average default rate for the equivalent agency grade - not a model estimate.")
            g4.metric("Security cover", "n/a" if pd.isna(cover) else f"{cover:.2f}x")
            rel = rating["relative"]
            (st.warning if rel["bias"] else st.caption)(rel["text"])
            with st.expander("How the grade is built"):
                st.dataframe(pd.DataFrame(rating["steps"], columns=["Step", "Effect", "Grade"]).fillna(""), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame([{"Grade": g, "≈ Agency": e, "Score from": lo, "Indicative 1-yr PD": f"{p_:.2%}", "Meaning": mn}
                                           for lo, g, e, p_, mn in rp.GRADES]), hide_index=True, width="stretch")

        ycols = list(bm.columns)[-3:]
        snap_items = [("Revenue (Rs cr)", None, "rev"), ("EBITDA (Rs cr)", None, "ebitda"), ("Net profit (Rs cr)", None, "ni"),
                      ("Operating cash flow (Rs cr)", None, "cfo"), ("Current ratio", "current_ratio", None), ("Interest cover", "icr", None),
                      ("DSCR (proxy)", "dscr", None), ("Debt / Equity", "de", None), ("Net debt / EBITDA", "nd_ebitda", None),
                      ("Cash conversion", "cce", None)]
        srows = []
        for label, mk, fk in snap_items:
            vals_ = []
            for y in ycols:
                if mk:
                    vals_.append(fmtv(mk, bm.loc[mk, y]))
                else:
                    v = BC.fin.loc[y, fk] if (fk in BC.fin.columns and y in BC.fin.index) else np.nan
                    vals_.append("n/a" if pd.isna(v) else f"{v:,.0f}")
            srows.append((label, vals_))

        cmp_m = cmp_df if BC is A else mA
        cmp_lbl = cmp_name if BC is A else A.name
        pcmp_ = pB if BC is A else pA
        hist_b = ew_b["hist"] if ew_b else None
        news_ctx = None
        if ew_b and ew_b.get("news"):
            n_items = ew_b["news"]["items"]
            n_rats = nw.latest_ratings(n_items)
            m_rt_ = st.session_state.get(f"nw-man-rt-{BC.ticker}", "–")
            if m_rt_ and m_rt_ != "–":                      # a rating the analyst entered on the News & ratings tab wins
                m_ag_ = st.session_state.get(f"nw-man-ag-{BC.ticker}", "CRISIL")
                n_rats = [dict(agency=m_ag_, rating=m_rt_, outlook=st.session_state.get(f"nw-man-ol-{BC.ticker}"),
                               action="Entered by analyst", scale="national", date=datetime.now(timezone.utc))] + \
                         [r_ for r_ in n_rats if r_["agency"] != m_ag_]
            news_ctx = dict(signal=nw.news_signal(n_items),
                            ratings=[dict(r_, cr=nw.grade_index_of(r_["rating"]) if r_["scale"] == "national" else None)
                                     for r_ in n_rats][:4],
                            top=[it for it in n_items if it["tone"] == "negative" and it["severity"] in ("high", "medium")][:5])
        ctx = dict(name=BC.name, sector=sector, as_of=f"data to {BC.period_end or 'n/a'}, fetched {BC.fetched_at or 'n/a'}",
                   vals=bvals, flags=bflags, pillars=pb_, score=sc_b, rating=rating, snap=dict(cols=[f"FY{y}" for y in ycols], rows=srows),
                   req=fac.get("req"), scen=fac.get("scen"), cov_rows=fac.get("cov_rows"), benford=bfA if BC is A else bfB,
                   cmp_name=cmp_lbl, basis=basis_sel, ew=ew_ctx, news=news_ctx, security_cover=cover, override=override,
                   override_reason=ov_reason.strip(),
                   analyst=dict(name=an_name.strip(), purpose=purpose.strip(), collateral_type=ctype, collateral_value=cval,
                                management=mgmt, outlook=outlook, notes=notes.strip()))
        fac_now = dict(fac)

        def memo_with_charts():
            """Runs only when a download button is clicked (on its own thread), so drawing charts never slows the page."""
            charts = dict(pillars=rp.chart_pillars(pb_ or {}, BC.name, pcmp_, cmp_lbl),
                          trends=rp.chart_trends(bm, BC.name, cmp_m, cmp_lbl),
                          heatmap=rp.chart_heatmap(fac_now.get("grid"), (fac_now.get("req") or {}).get("target", 1.25),
                                                   fac_now.get("dm_list", []), fac_now.get("dr_list", [])),
                          dd=rp.chart_dd(hist_b))
            return rp.build_memo({**ctx, "charts": charts})

        blocks = rp.build_memo(ctx)                      # on-screen preview: text only
        md_text = rp.to_markdown(blocks)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download PDF", lambda: rp.to_pdf(memo_with_charts()), file_name=f"credit_note_{BC.ticker}.pdf",
                           mime="application/pdf", key="dl-memo-pdf", type="primary", width="stretch")
        d2.download_button("Download Word", lambda: rp.to_docx(memo_with_charts()), file_name=f"credit_note_{BC.ticker}.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl-memo-docx", width="stretch")
        d3.download_button("Download Markdown", md_text, file_name=f"credit_note_{BC.ticker}.md", mime="text/markdown",
                           key="dl-memo", width="stretch")
        st.caption("The PDF and Word versions add four charts: factor scores, key-ratio trends, the DSCR stress heatmap and the "
                   "distance-to-default trend.")
        with st.container(border=True):
            st.markdown(md_text)

# ------------------------------------------------------------------ Insights
with tabs[9]:
    st.subheader("Insights & analysis")

    def render(secs):
        for head, bullets in secs:
            st.markdown(f"#### {head}")
            for b in bullets:
                st.markdown(f"- {b}")

    ic = A if B is None else {A.name: A, B.name: B}[st.radio("Analyse", [A.name, B.name], horizontal=True, key="ins-co")]
    is_A = ic is A
    i_vals, i_m, i_flags, i_bf, i_col = (vA, mA, flagsA, bfA, C1) if is_A else (vB, cmp_df, flagsB, bfB, C2)
    i_bench, i_bench_lbl, i_bench_m = (vB, cmp_name, cmp_df) if is_A else (vA, A.name, mA)
    i_tr = ins.trends(i_m, {t: m for t, m in all_m.items() if t != ic.ticker})
    i_rc = ins.root_causes(ic.fin, i_m)
    i_fix, i_cush = ins.fix_targets(ic.fin)
    i_cards = ins.cards(ic.name, i_vals, i_m, ic.fin, i_flags, i_tr, i_rc, i_bench, i_bench_lbl, latest_all)
    n_sev = {s: sum(1 for c in i_cards if c["severity"] == s) for s in ins.SEV_ORDER}
    v_ = ins.verdict(i_vals, i_flags)
    st.markdown(f"<div class='banner' style='--c:{SIGNAL_COL[v_['signal']]}'><div class='who' style='color:{i_col}'>● {esc(ic.name)}</div>"
                f"<div class='sig'>{v_['signal']}</div><div>{n_sev['High']} high-priority issues · {n_sev['Medium']} medium · "
                f"{n_sev['Strength']} strengths · compared with {esc(i_bench_lbl)}</div></div>", unsafe_allow_html=True)

    it = st.tabs(["Key insights", "Trends", "Root causes", "Peer position", "What it would take", "Summary"])
    SEV_COL = {"High": RED, "Medium": AMBER, "Low": CB, "Strength": GREEN}

    with it[0]:
        st.caption("Ranked by priority. Each card shows the evidence, why a lender should care, what to ask, and where to look in the dashboard.")
        show = st.segmented_control("Show", ["Risks", "Strengths", "All"], default="Risks", key="ins-filter") or "Risks"
        sel = [c for c in i_cards if show == "All" or (show == "Strengths") == (c["severity"] == "Strength")]
        if not sel:
            st.info("Nothing to show for this filter.")
        for c in sel:
            with st.container(border=True):
                st.markdown(f"<span style='background:{SEV_COL[c['severity']]};color:#fff;font-size:.7rem;font-weight:700;"
                            f"padding:2px 8px;border-radius:999px'>{c['severity'].upper()}</span> "
                            f"<b style='font-size:1.02rem'>{esc(c['title'])}</b>", unsafe_allow_html=True)
                st.markdown(c["body"])
                if c["evidence"]:
                    st.caption(" · ".join(c["evidence"]))
                extra = []
                if c["why"] and c["why"] not in c["body"]:
                    extra.append(f"**Why it matters:** {c['why']}")
                if c["ask"]:
                    extra.append(f"**What to ask:** {c['ask']}")
                extra.append(f"*See the {c['tab']} tab.*")
                st.markdown("  \n".join(extra))

    with it[1]:
        worse = [t for t in i_tr if not t["improving"]]
        better_ = [t for t in i_tr if t["improving"]]
        st.caption(f"Change over the full history available ({len(i_m.columns)} fiscal years). Moves that change zone come first, "
                   "then moves that are the largest in the sector.")
        l, r = st.columns(2)
        with l:
            st.markdown("**Getting worse**")
            for t in worse[:8]:
                st.markdown(f"- {t['text']}")
            if not worse:
                st.write("No rated metric has deteriorated.")
        with r:
            st.markdown("**Getting better**")
            for t in better_[:8]:
                st.markdown(f"- {t['text']}")
            if not better_:
                st.write("No rated metric has improved.")
        movers = [t for t in i_tr if t["zone"] != 0][:4] or i_tr[:4]
        if movers:
            st.markdown("##### Biggest moves")
            mc = st.columns(2)
            for j, t in enumerate(movers):
                with mc[j % 2]:
                    sers = {ic.name: i_m.loc[t["key"]]}
                    if t["key"] in i_bench_m.index:
                        sers[i_bench_lbl] = i_bench_m.loc[t["key"]]
                    st.plotly_chart(trend(t["key"], sers, h=250), width="stretch", key=f"ins-tr-{t['key']}")

    with it[2]:
        st.caption("Why the key numbers moved, traced back to the statement lines (latest year vs the year before).")
        if not i_rc:
            st.success("No major deterioration that needs explaining in the latest year.")
        for c in sorted(i_rc, key=lambda c: ins.SEV_ORDER[c["severity"]]):
            with st.container(border=True):
                st.markdown(f"<span style='color:{SEV_COL[c['severity']]};font-weight:700'>●</span> <b>{esc(c['title'])}</b>",
                            unsafe_allow_html=True)
                st.markdown(c["text"])
                if c["evidence"]:
                    st.caption(" · ".join(c["evidence"]))

    with it[3]:
        top, bottom = ins.peer_position(i_vals, latest_all)
        st.caption(f"Percentile = share of the {len(latest_all)} companies loaded in {sector} that {ic.name} beats on each metric "
                   "(higher is always better).")

        def pp_table(rows):
            return pd.DataFrame([{"Metric": mt.M[k]["label"], "Value": fmtv(k, i_vals.get(k)), "Percentile": p} for k, p in rows])
        cfg = {"Percentile": st.column_config.ProgressColumn("Percentile", min_value=0, max_value=100, format="%.0f")}
        l, r = st.columns(2)
        for col_, rows_, head_ in ((l, top, "Top quartile (sector leader)"), (r, bottom, "Bottom quartile (sector laggard)")):
            with col_:
                st.markdown(f"**{head_}**")
                if rows_:
                    st.dataframe(pp_table(rows_), column_config=cfg, hide_index=True, width="stretch")
                else:
                    st.write("None.")
        if len(latest_all) < 5:
            st.caption("Few companies are loaded for this sector, so percentiles are coarse.")

    with it[4]:
        st.caption(f"Based on FY{ic.fin.index[-1]} statements. Targets are the healthy thresholds used across the dashboard; "
                   "each fix assumes everything else stays the same.")
        l, r = st.columns(2)
        with l:
            st.markdown("**To reach the healthy zone**")
            if not i_fix:
                st.success("All the tested metrics are already in the healthy zone.")
            for f_ in i_fix:
                with st.container(border=True):
                    st.markdown(f"**{mt.M[f_['key']]['label']}**: {fmtv(f_['key'], f_['now'])} → target {fmtv(f_['key'], f_['target'])}")
                    st.markdown(f_["text"])
        with r:
            st.markdown("**Cushion before the concern zone**")
            if not i_cush:
                st.write("No healthy metric to stress.")
            for c in i_cush:
                with st.container(border=True):
                    st.markdown(f"**{mt.M[c['key']]['label']}**: {fmtv(c['key'], i_vals.get(c['key']))}")
                    st.markdown(c["text"])
        st.caption("Use the Credit workbench tab to test combined shocks and a new facility.")

    with it[5]:
        st.caption("The original rule-based summary.")
        render(ins.build(ic.name, i_vals, i_flags, cmp_label if is_A else A.name, i_bench, None if (B is None) else i_bench_lbl, i_bf))

# ------------------------------------------------------------------ Data & method
with tabs[10]:
    st.subheader("Data quality")
    dq = []
    for comp in ([A] + ([B] if B else [])):
        fin = comp.fin
        ok = int(all_m[comp.ticker].iloc[:, -1].notna().sum())
        gap = np.nan
        if {"ta", "tl", "equity"} <= set(fin.columns):
            r = fin.iloc[-1]
            gap = abs(r["ta"] - (r["tl"] + r["equity"])) / r["ta"] if r["ta"] else np.nan
        dq.append({"Company": comp.name, "Latest period": comp.period_end or "n/a", "Fetched": comp.fetched_at or "n/a",
                   "Years of data": len(fin), "Metrics computable": f"{ok} of {len(mt.M)}",
                   "Balance check": "n/a" if pd.isna(gap) else (f"OK ({gap:.1%} gap)" if gap < 0.05 else f"Check ({gap:.1%} gap - minority interest or Yahoo mapping)"),
                   "Fields Yahoo did not supply": ", ".join(comp.missing) or "none"})
    st.dataframe(pd.DataFrame(dq), width="stretch", hide_index=True)
    st.subheader("Data used")
    for comp in ([A] + ([B] if B else [])):
        st.write(f"Standardised annual statements for **{comp.name}** (Rs crore):")
        st.dataframe(comp.fin.T.style.format("{:,.1f}", na_rep="n/a"), width="stretch")
    st.subheader("Metric dictionary and thresholds")
    if SECTOR_MODE:
        st.caption(f"Cut-offs below are **sector-relative** for {sector} ({strict} strictness): taken from this sector's own "
                   "distribution where at least 6 companies have the metric, otherwise the textbook cut-off.")
    elif strict != "Standard":
        st.caption(f"Cut-offs below reflect the **{strict}** strictness chosen in View options (Standard ± 20%).")
    st.dataframe(pd.DataFrame([{"Metric": m["label"], "Group": m["group"], "Formula": m["formula"], "Meaning": m["desc"],
                                "Better when": {"high": "higher", "low": "lower", None: "-"}[m["dir"]],
                                "Healthy at": mt.thresholds(k)[0], "Concern at": mt.thresholds(k)[1],
                                "Cut-off source": "sector" if mt.is_sector_band(k) else ("textbook" if m["dir"] and m["green"] is not None else "-"),
                                "Textbook healthy": m["green"], "Textbook concern": m["red"]} for k, m in mt.M.items()]),
                 width="stretch", hide_index=True)
    st.subheader("Assumptions & limitations")
    st.markdown(f"""
- **Yahoo Finance gives about 4 annual periods**, so averages, Piotroski, Beneish and the Ohlson change term need the prior year and start in year 2. Data is cached for 6 hours; annual statements change only when a company files new results, while market cap moves daily.
- **DSCR is a proxy**: (CFO + interest) ÷ (interest + current portion of debt). True DSCR needs the repayment schedule.
- **Defensive interval** uses cash opex = Revenue − EBITDA; liquid assets = cash & short-term investments + receivables.
- **Cash-flow accrual** = (NI − CFO) ÷ average net operating assets; NOA = (Assets − cash) − (Liabilities − debt).
- **Altman Z (original)** uses *current* market cap, so only the latest year is shown. Z′ and Z″ use book values.
- **Beneish**: if SG&A or another index is missing, that index is set to neutral (1.0); results are then indicative. Not valid for banks/insurers (none are in the sector list).
- **Ohlson** size term depends on units: total assets in Rs crore ÷ price index (default 100).
- **Lifecycle stage** follows the sign pattern of operating / investing / financing cash flow.
- {ADV} **Facility & stress** is a one-year simplified pro-forma: revenue and margin shocks flow to EBITDA; interest scales with the rate change; working-capital change comes from receivable days and revenue-scaled stock and payables; tax at the company's recent rate. New debt proceeds are assumed spent in the business. Debt capacity ignores the tax shield on new interest, so it is conservative.
- {ADV} **Covenants, scorecard, watchlist, percentile ranks and the credit memo** are analyst aids built on the same metrics. The scorecard blends factor scores with adjustable weights and is not an agency rating.
- **Not included**: Montier C-score, Kaplan-Zingales, Fulmer H, abnormal-accrual regressions, LLCR and time-adjusted current ratio (these need repayment schedules or receivable ageing).
- Industry average = the companies listed for that sector in `sectors.py` (excluding the selected company), not the whole market. Only size-neutral numbers (ratios, margins, growth, days, scores) are averaged; rupee amounts such as revenue, profit and free cash flow are shown as a rank within the sector instead, because a median of very different-sized companies is not a benchmark.
- **Scoring basis.** *Textbook* applies the course cut-offs to every industry, which marks whole sectors as weak (e.g. most
  utilities fail Springate and Altman; most IT and pharma firms carry 'too many' working-capital days). *Sector-relative*
  judges each metric against the selected sector: healthy = at least the sector median, concern = its worst 25% (strictness
  shifts these percentiles). Hard floors stay: DSCR or interest cover below 1x and Ohlson probability above 50% are always a
  concern; Beneish and negative free cash flow keep their fixed rules. A metric with fewer than 6 sector values keeps the
  textbook cut-off.
- {ADV} **Early warning** uses Yahoo's quarterly income statements (about 5-7 quarters; Indian companies publish no quarterly cash-flow statement), the half-yearly interim balance sheet and two years of daily prices. Year-on-year compares the same quarter a year apart. **Merton model**: equity = call option on assets; market cap and one-year share-price volatility solve for asset value and volatility; default point = short-term debt + half of long-term debt (KMV); risk-free rate editable (default 6.5%); probability of default = N(−DD), risk-neutral and only indicative. The DD history holds debt at today's level. Early-warning signals always use the latest data, whatever the as-of year.
- {ADV} **News & ratings** runs three Google News searches per company (debt and ratings; legal and governance; rating agencies) plus Yahoo Finance's feed, keeps headlines from the last 12 months that name the company, and drops duplicates. A **keyword screen** tags each with a category, a tone for lenders and a severity; share-price chatter and broker calls (\"downgraded to Sell\") are treated as neutral. **News flow signal**: Concern only for a rating-agency downgrade in the last 180 days; Watch if a high-severity negative headline (default, insolvency, fraud, asset seizure, ban, auditor exit…) appeared in the last 180 days or three or more medium negatives in 90. A keyword match alone never makes a Concern, because namesakes (\"Cesc\" vs CESC) and stories where the company is the victim of a fraud slip through; short capitalised names must match in capitals. Testing on all 203 companies, only about 1 in 6 high-severity keyword hits was a genuine credit event. **Ratings** are read from headlines naming CRISIL, ICRA, CARE, India Ratings, Acuite, Brickwork, Moody's, Fitch or S&P, only from the clause that names the company. National-scale ratings map to CR grades (AAA→CR1, AA→CR2, A→CR3, BBB→CR4, BB→CR5, B+→CR6, B/B-→CR7, CCC→CR8, below→CR9); global-scale ratings are shown but not compared, since they sit under India's sovereign ceiling. The **AI read** (optional, needs an Anthropic API key) sends the non-chatter headlines to Claude, which returns a per-headline judgement, rating actions, a news-risk level, a digest and questions for management. It never changes the grade or the early-warning signal.
- {ADV} **Credit memo and internal grade.** The quantitative score on *textbook* cut-offs maps to a 9-step internal scale (CR1 ≈ AAA … CR9 ≈ CC/C) with an indicative one-year PD - a rounded long-run average default rate for the equivalent agency grade, not a company-specific estimate. Whole-notch adjustments: early warning (Watch −1, Elevated −2), management (Strong +1, Weak −1), negative industry outlook (−1), capped at ±2 together; security cover of 1.5x or more lifts the facility (not the issuer) one notch. The grade never depends on the screen's scoring basis; the sector-relative score is shown beside it as relative standing, with a warning when a weak grade mainly reflects the sector (e.g. utilities). PDF and Word files are built only when a download button is clicked.
- **Red-flag materiality.** Receivable, inventory and debt growth flags only fire when that balance is at least 2%, 2% and 5%
  of total assets respectively.
- **Fiscal-year labels** follow the Indian April-March year: an October-December year end is labelled with the next calendar
  year (Dec-2025 → FY2026), since it overlaps that fiscal year for 9 of 12 months.
- **Growth & margins** (revenue growth, EBITDA, net and free-cash-flow margins) are compared with peers but not rated green/red, because normal margins differ widely between sectors.
- Insights and the memo are rule-based text, not a credit rating or a lending decision.""", unsafe_allow_html=True)
