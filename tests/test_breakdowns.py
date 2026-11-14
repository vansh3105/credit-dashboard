"""The 'how is it calculated' breakdowns must reproduce the scores in metrics.compute, and the
insight rules must point at the right cause on a firm built to deteriorate."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import metrics as mt
import insights as ins


@pytest.fixture
def stressed():
    """Three years of a firm whose receivables balloon, borrowing climbs and cash flow turns negative."""
    return pd.DataFrame({
        "rev": [1000, 1100, 1150], "cogs": [600, 700, 780], "ebit": [150, 120, 60], "ebitda": [200, 170, 110],
        "ni": [80, 60, 10], "pbt": [110, 80, 15], "interest": [40, 50, 70], "tax": [30, 20, 5], "da": [50, 50, 50],
        "sga": [100, 110, 120], "ta": [1500, 1700, 1900], "ca": [600, 700, 750], "cl": [400, 550, 700], "cash": [100, 60, 30],
        "recv": [200, 300, 420], "inv": [150, 200, 260], "pay": [100, 90, 80], "debt": [500, 650, 850], "ltd": [400, 480, 600],
        "cur_debt": [100, 170, 250], "equity": [700, 720, 720], "retained": [300, 320, 330], "tl": [800, 980, 1180],
        "ppe": [700, 750, 800], "cfo": [120, 40, -30], "cfi": [-80, -90, -100], "capex": [-80, -90, -100],
        "shares": [1e8, 1e8, 1.1e8]}, index=[2023, 2024, 2025])


def test_breakdowns_match_compute(stressed):
    m = mt.compute(stressed)
    for y in stressed.index[1:]:
        b = mt.beneish_detail(stressed, y)
        assert b["m"] == pytest.approx(m.at["beneish", y])
        assert b["const"] + sum(r["contribution"] for r in b["rows"]) == pytest.approx(b["m"])
        p = mt.piotroski_detail(stressed, y)
        assert p["score"] == m.at["piotroski", y]
        o = mt.ohlson_detail(stressed, y)
        assert o["p"] == pytest.approx(m.at["ohlson_p", y])
    assert mt.beneish_detail(stressed, 2023) is None          # no prior year


def test_beneish_dsri_flags_receivable_build_up(stressed):
    rows = {r["key"]: r for r in mt.beneish_detail(stressed)["rows"]}
    assert rows["DSRI"]["index"] == pytest.approx((420 / 1150) / (300 / 1100))
    assert rows["DSRI"]["suspicious"]


def test_root_causes_find_the_drivers(stressed):
    rc = {c["key"]: c for c in ins.root_causes(stressed, mt.compute(stressed))}
    assert "receivable days" in rc["cce"]["text"]
    assert "weaker operating profit" in rc["icr"]["text"]           # EBIT -50% outweighs interest +40%
    assert rc["icr"]["severity"] == "High"
    assert "nd_ebitda" in rc and "current_ratio" in rc


def test_fix_targets_reach_the_threshold(stressed):
    fixes = {f["key"]: f for f in ins.fix_targets(stressed)[0]}
    nd, ebitda = 850 - 30, 110
    repay = nd - mt.M["nd_ebitda"]["green"] * ebitda
    assert f"Rs {repay:,.0f} cr" in fixes["nd_ebitda"]["text"]
    assert "refinancing alone cannot fix" in fixes["dscr"]["text"]    # CFO + interest < 0 at 1.5x target


def test_cards_rank_risks_first(stressed):
    m = mt.compute(stressed)
    cards = ins.cards("X", ins.latest(m), m, stressed, mt.red_flags(stressed, m), ins.trends(m), ins.root_causes(stressed, m))
    order = [ins.SEV_ORDER[c["severity"]] for c in cards]
    assert order == sorted(order) and cards[0]["severity"] == "High"


def test_growth_and_margins(stressed):
    m = mt.compute(stressed)
    assert m.at["rev_growth", 2025] == pytest.approx(1150 / 1100 - 1)
    assert m.at["ebitda_margin", 2025] == pytest.approx(110 / 1150)
    assert m.at["fcf_margin", 2025] == pytest.approx((-30 - 100) / 1150)
    assert mt.rag("ebitda_margin", 0.05) is None                    # compared, not rated
    assert ins.better("ebitda_margin", 0.2, 0.1) is True


def test_rupee_amounts_get_no_sector_percentile():
    assert np.isnan(mt.percentile("fcf", 500, [100, 200, 300, 400]))   # would only rank company size
    assert mt.percentile("net_margin", 0.25, [0.1, 0.2, 0.3, 0.4]) == 50.0
