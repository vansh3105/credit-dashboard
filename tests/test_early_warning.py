"""Early-warning maths: Merton against Hull's worked example, quarterly YoY matching, and the signal rules."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import early_warning as ew


def test_merton_matches_hull_example():
    # Hull, Options Futures & Other Derivatives: E=3, sigma_E=80%, D=10, r=5%, T=1 -> V=12.40, sigma_V=21.23%, PD=12.7%
    r = ew.merton(3, 0.80, 10, r=0.05, T=1)
    assert r["V"] == pytest.approx(12.40, abs=0.01)
    assert r["sigma_V"] == pytest.approx(0.2123, abs=0.0005)
    assert r["PD"] == pytest.approx(0.127, abs=0.001)
    assert ew.merton(100, 0.3, 0)["no_debt"] and ew.merton(100, 0.3, 0)["PD"] == 0
    assert ew.default_point(20, 100) == 60                    # 20 short-term + half of 80 long-term


def test_quarterly_yoy_pairs_same_quarter():
    q = pd.DataFrame({"rev": [100, 110, 120, 130, 90], "ebitda": [20, 22, 24, 26, 9], "ebit": [15] * 5, "interest": [5] * 5,
                      "ni": [5, 5, 5, 5, -2]},
                     index=pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]))
    t = ew.quarterly_table(q)
    assert t["Revenue YoY"].iloc[-1] == pytest.approx(-0.10)          # Jun-26 vs Jun-25, not vs Mar-26
    assert np.isnan(t["Revenue YoY"].iloc[0])
    assert t["Margin change YoY"].iloc[-1] == pytest.approx(0.10 - 0.20)
    assert ew.ttm(q, "ebitda") == pytest.approx(22 + 24 + 26 + 9)
    sig = {s["signal"].split(" (")[0]: s["status"] for s in ew.signals(pd.DataFrame(), {"is": q}, None, [], None, None)}
    assert sig["Revenue vs same quarter last year"] == "A"            # -10% exactly: watch, not concern
    assert sig["EBITDA margin vs same quarter last year"] == "R"
    assert sig["Loss-making quarters"] == "R"                         # latest quarter is a loss


def test_interim_balance_sheet_not_compared_with_itself():
    fin = pd.DataFrame({"debt": [100.0], "cash": [50.0], "ta": [1000.0], "ebitda": [40.0]}, index=[2026])
    bs = pd.DataFrame({"debt": [100.0], "cash": [50.0]}, index=pd.to_datetime(["2026-03-31"]))
    sigs = ew.signals(fin, {"is": pd.DataFrame(), "bs": bs}, None, [], None, None, pd.Timestamp("2026-03-31"))
    assert not any("since the last annual report" in s["signal"] for s in sigs)
    bs2 = pd.DataFrame({"debt": [140.0], "cash": [20.0]}, index=pd.to_datetime(["2026-09-30"]))
    sigs = {s["signal"].split(" (")[0]: s["status"] for s in ew.signals(fin, {"is": pd.DataFrame(), "bs": bs2}, None, [], None, None,
                                                                         pd.Timestamp("2026-03-31"))}
    assert sigs["Total debt since the last annual report"] == "R"     # +40%, debt 10% of assets
    assert sigs["Cash since the last annual report"] == "R"           # -60%


def test_market_signals_against_sector():
    mkt = dict(ret_1y=-0.35, off_high=-0.45, vol=0.60, max_dd=-0.5, last=1)
    peers = [dict(ret_1y=0.05, off_high=-0.1, vol=0.30, max_dd=-0.2, last=1)] * 5
    s = {x["signal"]: x["status"] for x in ew.signals(pd.DataFrame(), None, mkt, peers, None, None)}
    assert s["Share price vs 52-week high"] == "R"
    assert s["1-year share return vs sector median"] == "R"          # 40 pp behind
    assert s["Share-price volatility vs sector median"] == "R"       # 2x the sector


def test_cash_fall_capped_when_still_net_cash_and_dd_trend_needs_low_level():
    fin = pd.DataFrame({"debt": [10.0], "cash": [500.0], "ta": [1000.0]}, index=[2026])
    bs = pd.DataFrame({"debt": [10.0], "cash": [100.0]}, index=pd.to_datetime(["2026-09-30"]))
    s = {x["signal"].split(" (")[0]: x["status"] for x in ew.signals(fin, {"bs": bs}, None, [], None, None, pd.Timestamp("2026-03-31"))}
    assert s["Cash since the last annual report"] == "A"              # -80% but cash still 10x debt
    mert = dict(no_debt=False, PD=0.0, DD=15.0)
    high = pd.DataFrame({"DD": [20.0] * 6 + [15.0]})                  # big fall from a very safe level
    low = pd.DataFrame({"DD": [5.5] * 6 + [3.0]})
    get = lambda h: {x["signal"]: x["status"] for x in ew.signals(pd.DataFrame(), None, None, [], mert, h)}["Distance-to-default, change over 6 months"]
    assert get(high) == "G" and get(low) == "R"
