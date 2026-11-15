"""Score attribution, what-if engine and break points, bank/NBFC ratios, groups, pledge notch, data gap fills."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import banks as bk
import data as dt
import groups
import metrics as mt
import report as rp
import scenario as sc
import sectors

FIN = pd.DataFrame({
    "rev": [1000.0, 1100.0], "cogs": [600.0, 650.0], "ebitda": [200.0, 220.0], "ebit": [150.0, 165.0], "da": [50.0, 55.0],
    "interest": [40.0, 44.0], "pbt": [115.0, 126.0], "tax": [29.0, 32.0], "ni": [86.0, 94.0], "ta": [1500.0, 1600.0],
    "ca": [600.0, 650.0], "cl": [400.0, 420.0], "cash": [100.0, 110.0], "recv": [200.0, 220.0], "inv": [150.0, 160.0],
    "pay": [120.0, 130.0], "debt": [500.0, 520.0], "cur_debt": [100.0, 100.0], "equity": [700.0, 760.0], "tl": [800.0, 840.0],
    "cfo": [160.0, 175.0], "retained": [300.0, np.nan], "divs": [-20.0, -25.0]}, index=[2025, 2026])


def test_score_attribution_adds_up():
    mt.set_strictness("Standard"); mt.set_basis("Textbook")
    m = mt.compute(FIN)
    v_now = pd.Series({k: m.loc[k].iloc[-1] for k in m.index})
    v_prev = v_now.copy()
    v_prev["dscr"], v_prev["icr"] = 0.5, 1.0                              # last year: cover in the concern zone
    a = mt.score_attribution(v_now, v_prev)
    assert a["delta"] > 0 and {x["key"] for x in a["items"]} >= {"dscr", "icr"}
    assert abs(a["delta"] - sum(x["impact"] for x in a["items"]) - a["other"]) < 1e-9
    assert all(x["impact"] > 0 for x in a["items"] if x["key"] in ("dscr", "icr"))


def test_whatif_and_breakpoints():
    r = sc.shock_row(FIN.iloc[-1], g=-0.2, dm=-0.02, dr=0.01)
    assert r["rev"] == 880.0 and abs(r["ebitda"] - 880 * (220 / 1100 - 0.02)) < 1e-9
    assert abs(r["interest"] - (44 + 520 * 0.01)) < 1e-9 and r["recv"] == 220 * 0.8
    same = sc.run(FIN)
    assert same["score0"] == same["score1"]                               # no shock, no change
    worse = sc.run(FIN, g=-0.4, dm=-0.05)
    assert worse["shocked"]["icr"] < worse["base"]["icr"] and worse["score1"] <= worse["score0"]
    bp = sc.breakpoints(FIN)
    assert bp["DSCR < 1.0x"] is None or bp["DSCR < 1.25x"] <= bp["DSCR < 1.0x"]


def test_retained_roll_forward_and_zero_inventory():
    fin, missing = dt.structural_zeros(FIN.drop(columns=["inv"]), ["inv"])
    assert fin["inv"].eq(0).all() and "inv" not in missing
    assert fin.loc[2026, "retained"] == 300 + 94 - 25                     # RE + NI - dividends
    no_div = FIN.drop(columns=["divs"])
    assert pd.isna(dt.structural_zeros(no_div, [])[0].loc[2026, "retained"])   # unknown dividend: no guess


def test_bank_ratios_and_score():
    f = pd.DataFrame({"ta": [1000.0, 1100.0], "equity": [100.0, 110.0], "loans": [600.0, 660.0], "debt": [100.0, 110.0],
                      "tl": [900.0, 990.0], "rev": [60.0, 66.0], "ii": [80.0, 88.0], "ie": [50.0, 54.0], "opex_op": [25.0, 27.0],
                      "pbt": [25.0, 30.0], "ni": [18.0, 22.0]}, index=[2025, 2026])
    v = bk.latest(bk.compute(f))
    assert abs(v["roa"] - 22 / 1050) < 1e-12 and abs(v["nim"] - 34 / 1050) < 1e-12
    assert abs(v["credit_cost"] - (66 - 27 - 30) / 630) < 1e-12 and abs(v["cost_income"] - 27 / 66) < 1e-12
    p = bk.pillar_scores(v, "bank")
    s0 = bk.score(p)
    v2 = bk.with_manual(v, {"gnpa": 8.0, "nnpa": 4.0, "crar": 10.0})       # bad NPAs, thin capital
    assert bk.rag("gnpa", v2["gnpa"], "bank") == "R" and bk.score(bk.pillar_scores(v2, "bank")) < s0
    assert bk.signal(v2, "bank") == "ELEVATED CONCERN"
    bad = f.assign(ii=[5.0, 6.0])                                          # interest lines incomplete -> blanks, with a note
    mb = bk.compute(bad)
    assert pd.isna(bk.latest(mb)["nim"]) and mb.attrs["notes"]


def test_sectors_groups_and_pledge():
    assert set(sectors.FIN_SECTORS) <= set(sectors.SECTORS) and sectors.FIN_KIND["PFC.NS"] == "nbfc"
    allt = {t for d in sectors.SECTORS.values() for t in d}
    assert all(t in allt for ts in groups.GROUPS.values() for t in ts)
    assert groups.group_of("TATASTEEL.NS") == "Tata" and groups.group_of("INFY.NS") is None
    assert rp.rate(71, None, "Adequate", "Stable", None, 55)["issuer"]["grade"] == "CR5"
    assert rp.rate(71, None, "Adequate", "Stable", None, 30)["issuer"]["grade"] == "CR4"


def test_alert_report():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("alerts", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "alerts.py")
    al = importlib.util.module_from_spec(spec); spec.loader.exec_module(al)
    h = pd.DataFrame([{"date": "2026-11-14", "ticker": "X.NS", "name": "X", "grade": "CR4", "score": 70.0, "signal": "COMFORTABLE",
                       "ew_level": "COMFORTABLE", "news": "G"},
                      {"date": "2026-11-15", "ticker": "X.NS", "name": "X", "grade": "CR5", "score": 62.0, "signal": "WATCH",
                       "ew_level": "COMFORTABLE", "news": "G"}])
    h["date"] = pd.to_datetime(h["date"])
    md, worse = al.report(h, ["X.NS", "NOPE.NS"])
    assert worse == 3 and "CR4 | CR5" in md and "not in the dashboard" in md
