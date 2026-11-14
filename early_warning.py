"""Early-warning signals: what has changed since the last annual report.

Three sources, each fresher than the annual statements the rest of the dashboard uses:
  * quarterly results (income statement) and the interim, usually half-yearly, balance sheet;
  * the share price (fall from peak, return and volatility versus sector peers);
  * the Merton (1974) structural model: equity is a call option on the firm's assets, so market cap and share-price
    volatility imply the asset value, asset volatility and the distance to the default point (KMV convention:
    short-term debt + half of long-term debt). Distance-to-default (DD) -> probability of default = N(-DD).
Money in Rs crore. Every rule is listed with its signal so the reading can be checked.
"""
import math

import numpy as np
import pandas as pd

import metrics as mt

TRADING_DAYS = 252


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------ Merton / KMV
def default_point(cur_debt, total_debt):
    """KMV default point: short-term debt + half of long-term debt."""
    if total_debt is None or pd.isna(total_debt):
        return np.nan
    cur = 0.0 if cur_debt is None or pd.isna(cur_debt) else min(cur_debt, total_debt)
    return cur + 0.5 * (total_debt - cur)


def merton(E, sigma_E, D, r=0.065, T=1.0):
    """Solve E = V N(d1) - D e^{-rT} N(d2) and sigma_E E = N(d1) sigma_V V for V and sigma_V.
    Returns dict(V, sigma_V, DD, PD, D, E, sigma_E, no_debt) or None when inputs are missing."""
    if any(x is None or pd.isna(x) for x in (E, sigma_E, D)) or E <= 0 or sigma_E <= 0:
        return None
    if D <= 0:
        return dict(V=E, sigma_V=sigma_E, DD=np.inf, PD=0.0, D=D, E=E, sigma_E=sigma_E, no_debt=True)
    sq = math.sqrt(T)
    V, sV = E + D, sigma_E * E / (E + D)
    for _ in range(500):
        for _ in range(100):                           # Newton step on V for the current sigma_V
            d1 = (math.log(V / D) + (r + 0.5 * sV * sV) * T) / (sV * sq)
            f = V * ncdf(d1) - D * math.exp(-r * T) * ncdf(d1 - sV * sq) - E
            Vn = V - f / max(ncdf(d1), 1e-12)
            Vn = Vn if Vn > 0 else V / 2
            done = abs(Vn - V) < 1e-10 * V
            V = Vn
            if done:
                break
        d1 = (math.log(V / D) + (r + 0.5 * sV * sV) * T) / (sV * sq)
        sVn = sigma_E * E / (V * max(ncdf(d1), 1e-12))
        if abs(sVn - sV) < 1e-10:
            sV = sVn
            break
        sV = sVn
    DD = (math.log(V / D) + (r - 0.5 * sV * sV) * T) / (sV * sq)
    return dict(V=V, sigma_V=sV, DD=DD, PD=ncdf(-DD), D=D, E=E, sigma_E=sigma_E, no_debt=False)


# ------------------------------------------------------------------ market statistics
def market_stats(px):
    """1-year return, fall from the 52-week high, annualised volatility and worst drawdown, from daily closes."""
    px = px.dropna()
    if len(px) < 60:
        return None
    y = px.iloc[-TRADING_DAYS:]
    lr = np.log(y / y.shift(1)).dropna()
    return dict(ret_1y=y.iloc[-1] / y.iloc[0] - 1, off_high=y.iloc[-1] / y.max() - 1,
                vol=float(lr.std() * math.sqrt(TRADING_DAYS)), max_dd=float((y / y.cummax() - 1).min()), last=y.iloc[-1])


def dd_history(px, mcap_now, D, r=0.065, months=12):
    """Distance-to-default at each month end over the last `months`, holding debt at today's level and scaling market
    cap with the share price. Volatility = trailing one-year daily volatility at that date."""
    px = px.dropna()
    if len(px) < TRADING_DAYS + 30 or pd.isna(mcap_now) or pd.isna(D):
        return pd.DataFrame()
    lr = np.log(px / px.shift(1))
    vol = lr.rolling(TRADING_DAYS).std() * math.sqrt(TRADING_DAYS)
    ends = px.groupby([px.index.year, px.index.month]).tail(1).index[-(months + 1):]
    rows = []
    for d in ends:
        res = merton(mcap_now * px[d] / px.iloc[-1], vol[d], D, r)
        if res and not res["no_debt"]:
            rows.append((d, res["DD"], res["PD"]))
    return pd.DataFrame(rows, columns=["date", "DD", "PD"]).set_index("date")


# ------------------------------------------------------------------ quarterly view
def _year_ago(idx, q):
    """The quarter in `idx` closest to one year before q (within 25 days), or None."""
    target = q - pd.DateOffset(years=1)
    near = [x for x in idx if abs((x - target).days) <= 25]
    return near[0] if near else None


def quarterly_table(q_is):
    """Per quarter: revenue, YoY revenue growth, EBITDA margin, margin change vs the same quarter last year,
    interest cover, net income."""
    if q_is is None or q_is.empty or "rev" not in q_is:
        return pd.DataFrame()
    t = pd.DataFrame(index=q_is.index)
    g = lambda k: q_is[k] if k in q_is else pd.Series(np.nan, index=q_is.index)
    t["Revenue"] = g("rev")
    t["EBITDA margin"] = g("ebitda") / g("rev")
    t["Interest cover"] = (g("ebit") / g("interest")).where(g("interest") > 0)
    t["Net income"] = g("ni")
    yoy, dm = [], []
    for q in t.index:
        p = _year_ago(t.index, q)
        yoy.append(t.at[q, "Revenue"] / t.at[p, "Revenue"] - 1 if p is not None and t.at[p, "Revenue"] else np.nan)
        dm.append(t.at[q, "EBITDA margin"] - t.at[p, "EBITDA margin"] if p is not None else np.nan)
    t["Revenue YoY"] = yoy
    t["Margin change YoY"] = dm
    return t.replace([np.inf, -np.inf], np.nan)


def ttm(q_is, col):
    """Sum of the last four quarters if they are consecutive (span of about nine months), else NaN."""
    if q_is is None or col not in q_is or len(q_is) < 4:
        return np.nan
    last4 = q_is[col].iloc[-4:]
    span = (last4.index[-1] - last4.index[0]).days
    return float(last4.sum()) if 250 <= span <= 300 and last4.notna().all() else np.nan


# ------------------------------------------------------------------ signals
def _sig(area, name, value, rule, status, why):
    return dict(area=area, signal=name, value=value, rule=rule, status=status, why=why)


def signals(fin, q, mkt, sector_mkt, mert, dd_hist, annual_end=None):
    """All early-warning signals for one company. fin: annual statements; q: get_quarterly() result;
    mkt: market_stats(); sector_mkt: list of market_stats() of sector peers; mert: merton(); dd_hist: dd_history().
    Status 'R' / 'A' / 'G', or None when the data is missing."""
    out = []
    pct = lambda v: "n/a" if v is None or pd.isna(v) else f"{v:+.1%}"
    qt = quarterly_table(q.get("is")) if q else pd.DataFrame()

    # --- quarterly results
    yo = qt["Revenue YoY"].dropna() if len(qt) else pd.Series(dtype=float)
    if len(yo):
        last, two_neg = yo.iloc[-1], len(yo) >= 2 and (yo.iloc[-2:] < 0).all()
        st = "R" if (last < -0.10 or two_neg) else ("A" if last < 0 else "G")
        out.append(_sig("Quarterly results", f"Revenue vs same quarter last year ({yo.index[-1]:%b-%y})", pct(last),
                        "Concern below −10% or two falling quarters in a row; watch if below 0%", st,
                        "Falling sales are usually the first thing to show up, months before the annual report."))
    dm = qt["Margin change YoY"].dropna() if len(qt) else pd.Series(dtype=float)
    if len(dm):
        v = dm.iloc[-1]
        out.append(_sig("Quarterly results", f"EBITDA margin vs same quarter last year ({dm.index[-1]:%b-%y})", f"{v * 100:+.1f} pp",
                        "Concern if down more than 3 pp; watch if down more than 1 pp", "R" if v < -0.03 else ("A" if v < -0.01 else "G"),
                        "A shrinking margin erodes the cash available for interest before revenue visibly falls."))
    ic = qt["Interest cover"].dropna() if len(qt) else pd.Series(dtype=float)
    if len(ic):
        v = ic.iloc[-1]
        out.append(_sig("Quarterly results", f"Interest cover, latest quarter ({ic.index[-1]:%b-%y})", f"{v:.2f}x",
                        "Same healthy / concern cut-offs as annual interest cover", mt.rag("icr", v),
                        "Quarterly EBIT ÷ interest: shows coverage slipping before the year-end numbers do."))
    if len(qt) and qt["Net income"].notna().any():
        ni = qt["Net income"].dropna().iloc[-4:]
        n_loss = int((ni < 0).sum())
        out.append(_sig("Quarterly results", "Loss-making quarters (last four)", f"{n_loss} of {len(ni)}",
                        "Concern if the latest quarter is a loss; watch if any of the last four is",
                        "R" if ni.iloc[-1] < 0 else ("A" if n_loss else "G"), "Losses eat into equity, the lender's cushion."))
    t_e = ttm(q.get("is") if q else None, "ebitda")
    a_e = fin["ebitda"].iloc[-1] if "ebitda" in fin and len(fin) else np.nan
    if not (pd.isna(t_e) or pd.isna(a_e)) and a_e > 0:
        ch = t_e / a_e - 1
        out.append(_sig("Quarterly results", "EBITDA, last four quarters vs last annual report", pct(ch),
                        "Concern if down more than 15%; watch if down more than 5%", "R" if ch < -0.15 else ("A" if ch < -0.05 else "G"),
                        "Rolls the annual picture forward: are the annual ratios already out of date?"))

    # --- interim balance sheet vs the last annual one
    bs = q.get("bs") if q else None
    if bs is not None and len(bs) and len(fin):
        latest = bs.index[-1]
        if annual_end is None or latest > annual_end + pd.Timedelta(days=20):
            row, fy = bs.iloc[-1], fin.iloc[-1].copy()
            if annual_end is not None:              # compare like with like: the year-end column of the same interim report
                same = [d for d in bs.index if abs((d - annual_end).days) <= 20]
                if same:
                    for k_ in ("debt", "cash", "ca", "cl"):
                        if k_ in bs and not pd.isna(bs.at[same[0], k_]):
                            fy[k_] = bs.at[same[0], k_]
            d0, d1 = fy.get("debt", np.nan), row.get("debt", np.nan)
            if not (pd.isna(d0) or pd.isna(d1)) and d0 > 0:
                ch = d1 / d0 - 1
                material = "ta" in fy and fy["ta"] > 0 and max(d0, d1) >= 0.05 * fy["ta"]
                st = (("R" if ch > 0.25 else ("A" if ch > 0.10 else "G")) if material else "G")
                out.append(_sig("Interim balance sheet", f"Total debt since the last annual report ({latest:%b-%y})", pct(ch),
                                "Concern if up more than 25%, watch if up more than 10% (only when debt is at least 5% of assets)",
                                st, "New borrowing between annual reports is a classic early sign of cash stress."))
            c0, c1 = fy.get("cash", np.nan), row.get("cash", np.nan)
            if not (pd.isna(c0) or pd.isna(c1)) and c0 > 0:
                ch = c1 / c0 - 1
                st = "R" if ch < -0.5 else ("A" if ch < -0.3 else "G")
                still_net_cash = not pd.isna(d1) and c1 >= d1
                if still_net_cash and st == "R":
                    st = "A"                                  # cash still covers all debt: worth a look, not a credit alarm
                out.append(_sig("Interim balance sheet", f"Cash since the last annual report ({latest:%b-%y})", pct(ch),
                                "Concern if down more than 50%; watch if down more than 30% (capped at watch while cash still exceeds total debt)",
                                st, "A falling cash pile shortens the runway." + (" Cash still exceeds total debt." if still_net_cash else "")))
            if not (pd.isna(row.get("ca", np.nan)) or pd.isna(row.get("cl", np.nan))) and row["cl"] > 0:
                v = row["ca"] / row["cl"]
                out.append(_sig("Interim balance sheet", f"Current ratio ({latest:%b-%y})", f"{v:.2f}x",
                                "Same cut-offs as the annual current ratio", mt.rag("current_ratio", v),
                                "Short-term liquidity at the latest half-year."))

    # --- market
    if mkt:
        v = mkt["off_high"]
        out.append(_sig("Market", "Share price vs 52-week high", pct(v), "Concern below −40%; watch below −25%",
                        "R" if v <= -0.40 else ("A" if v <= -0.25 else "G"),
                        "Equity investors react within days; a deep fall often precedes rating downgrades."))
        peers_r = [m["ret_1y"] for m in sector_mkt if m]
        if len(peers_r) >= 4:
            gap = mkt["ret_1y"] - float(np.median(peers_r))
            out.append(_sig("Market", "1-year share return vs sector median", f"{gap * 100:+.0f} pp",
                            "Concern if more than 30 pp behind; watch if more than 15 pp behind",
                            "R" if gap < -0.30 else ("A" if gap < -0.15 else "G"),
                            "Underperformance against peers points to company-specific trouble, not the sector cycle."))
        peers_v = [m["vol"] for m in sector_mkt if m]
        if len(peers_v) >= 4:
            ratio = mkt["vol"] / float(np.median(peers_v))
            out.append(_sig("Market", "Share-price volatility vs sector median", f"{ratio:.2f}x ({mkt['vol']:.0%} a year)",
                            "Concern above 1.5x the sector; watch above 1.2x", "R" if ratio > 1.5 else ("A" if ratio > 1.2 else "G"),
                            "Higher volatility means a wider range of outcomes - and a shorter distance to default."))
    if mert:
        if mert["no_debt"]:
            out.append(_sig("Merton model", "Probability of default, 1 year", "≈ 0% (no debt)", "Concern above 5%; watch above 1%", "G",
                            "With no debt there is no default point."))
        else:
            out.append(_sig("Merton model", "Probability of default, 1 year", f"{mert['PD']:.2%} (DD {mert['DD']:.1f})",
                            "Concern above 5%; watch above 1%", "R" if mert["PD"] > 0.05 else ("A" if mert["PD"] > 0.01 else "G"),
                            "Market-implied chance that asset value falls below the default point within a year."))
            if dd_hist is not None and len(dd_hist) >= 7:
                ch, now = dd_hist["DD"].iloc[-1] - dd_hist["DD"].iloc[-7], dd_hist["DD"].iloc[-1]
                st = "R" if (ch < -2 and now < 4) else ("A" if (ch < -1 and now < 6) else "G")
                out.append(_sig("Merton model", "Distance-to-default, change over 6 months", f"{ch:+.1f} (now {now:.1f})",
                                "Concern if down more than 2 and now below 4; watch if down more than 1 and now below 6 "
                                "(a fall from a very high level does not matter)", st,
                                "The trend matters as much as the level: a falling DD is the model's early warning."))
    return out


def summary(sigs):
    n = {s: sum(1 for x in sigs if x["status"] == s) for s in ("R", "A", "G")}
    level = "ELEVATED CONCERN" if n["R"] >= 3 else ("WATCH" if n["R"] or n["A"] >= 3 else "COMFORTABLE")
    return n, level
