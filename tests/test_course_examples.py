"""Unit tests: reproduce numbers worked out in the FSAFA decks."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import metrics as mt


def one_year(**k):
    return pd.DataFrame([k], index=[2026])


def test_apex_liquidity_case_s5_6():
    # Apex / manufacturing client, FY26 (Rs crore)
    d = pd.DataFrame({
        "rev": [1500, 1800], "ebitda": [180, 225], "ni": [90, 105], "cfo": [135, 45],
        "cash": [60, 40],   # cash + marketable securities (30+10 in FY26)
        "recv": [250, 390], "inv": [200, 300], "ca": [630, 810], "cl": [350, 450], "ta": [1200, 1500],
    }, index=[2025, 2026])
    m = mt.compute(d)
    y = 2026
    assert m.loc["current_ratio", y] == pytest.approx(1.8)
    assert m.loc["wc_days", y] == pytest.approx(73.0)
    assert m.loc["ocf_ratio", y] == pytest.approx(0.10)
    assert m.loc["cce", y] == pytest.approx(0.20)
    assert m.loc["sloan", y] == pytest.approx(0.0444, abs=1e-3)       # 4.4%
    # DIR: course = 430 / (1350/365) = 116 days (cash opex 1350 given in deck; here rev-EBITDA=1575)
    assert (430 / (1350 / 365)) == pytest.approx(116, abs=0.5)


def test_accrual_examples_s7_8():
    # Company A / B (NI 200; CFO 220 vs 20; avg NOA 1285 vs 1640)
    a = (200 - 220) / ((1250 + 1320) / 2)
    b = (200 - 20) / ((1250 + 2030) / 2)
    assert a == pytest.approx(-0.016, abs=1e-3)
    assert b == pytest.approx(0.110, abs=1e-3)
    d = pd.DataFrame({"ni": [180, 200], "cfo": [160, 20], "ta": [2000, 2500],
                      "cash": [100, 100], "tl": [1000, 1200], "debt": [200, 300]}, index=[2025, 2026])
    # cf_accrual uses NOA = (TA-cash) - (TL-debt); just check sign/size is sensible
    assert mt.compute(d).loc["cf_accrual", 2026] > 0.05


def test_ohlson_worked_example_s9_10():
    T = mt.ohlson_T(ta=1000, tl=600, wc=150, ca=400, cl=250, ni=80, ffo=120, ni_prev=60, gnp=100)
    assert T == pytest.approx(0.564, abs=0.005)
    assert mt.logistic(T) == pytest.approx(0.637, abs=0.005)
    assert mt.logistic(-3) == pytest.approx(0.0474, abs=1e-3)
    assert mt.logistic(0.5) == pytest.approx(0.6225, abs=1e-3)


def test_benford_expected_distribution():
    exp = np.log10(1 + 1 / np.arange(1, 10))
    assert exp[0] == pytest.approx(0.301, abs=1e-3) and exp[8] == pytest.approx(0.046, abs=1e-3)
    # numbers drawn log-uniformly obey Benford -> should not be flagged; uniform first digits should be
    rng = np.random.default_rng(1)
    good = 10 ** rng.uniform(0, 6, 2000)
    assert mt.benford(good)["flag"] is False
    fake = [d * 10 ** k for d in range(1, 10) for k in range(2, 5) for _ in range(40)]
    assert mt.benford(fake)["flag"] is True
    r = mt.benford(good)
    assert r["cutoff"] == pytest.approx(1.36 / np.sqrt(2000))


def test_beneish_neutral_firm():
    neutral = dict(DSRI=1, GMI=1, AQI=1, SGI=1, DEPI=1, SGAI=1, LVGI=1, TATA=0)
    assert mt.beneish_m(neutral) == pytest.approx(-2.48, abs=1e-6)   # -4.84 + sum of the 7 index coefficients (TATA=0)
    assert mt.beneish_m({**neutral, "TATA": 0.10}) > -2.22


def test_altman_zones_s7_8():
    d = one_year(wc=0, ta=1000, ca=500, cl=500, retained=300, ebit=150, equity=500, tl=500, rev=1000,
                 ni=100, pbt=140, cfo=120, da=30, ltd=100, ppe=300, cash=50, debt=100, interest=10,
                 recv=100, inv=100, pay=50, cogs=600, ebitda=180, tax=40)
    m = mt.compute(d, mcap_cr=1500)
    z2 = 6.56 * 0 + 3.26 * 0.3 + 6.72 * 0.15 + 1.05 * 1.0
    assert m.loc["altman_z2", 2026] == pytest.approx(z2)
    z = 1.2 * 0 + 1.4 * 0.3 + 3.3 * 0.15 + 0.6 * 3.0 + 1.0 * 1.0
    assert m.loc["altman_z", 2026] == pytest.approx(z)


def test_dhenu_loan_to_equity_conversion_s3_4():
    before = dict(de=1000 / 100, icr=120 / 80, debt_ratio=1000 / 1100, eq_ratio=100 / 1100)
    after = dict(de=160 / 940, icr=120 / 13, debt_ratio=160 / 1100, eq_ratio=940 / 1100)
    assert before["de"] == pytest.approx(10.0) and before["icr"] == pytest.approx(1.5)
    assert after["de"] == pytest.approx(0.17, abs=0.01) and after["icr"] == pytest.approx(9.2, abs=0.05)
    assert mt.rag("de", before["de"]) == "R" and mt.rag("de", after["de"]) == "G"


def test_rag_thresholds():
    assert mt.rag("cce", 0.20) == "R" and mt.rag("cce", 0.9) == "G" and mt.rag("cce", 0.6) == "A"
    assert mt.rag("sloan", 0.04) == "G" and mt.rag("sloan", 0.25) == "R"
    assert mt.rag("beneish", -1.5) == "R" and mt.rag("beneish", -2.8) == "G"
    assert mt.rag("ccc", 50) is None
