"""Credit memo: grade mapping and notching, and that every renderer produces a valid file with the same content."""
import io, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import report as rp


def test_grade_scale_and_notches():
    assert rp.grade_index(95) == 0 and rp.grade_index(70) == 3 and rp.grade_index(10) == len(rp.GRADES) - 1
    r = rp.rate(71, "COMFORTABLE", "Adequate", "Stable", None)          # BBB-equivalent, no adjustments
    assert r["issuer"]["grade"] == "CR4" and r["facility"]["grade"] == "CR4"
    r = rp.rate(71, "ELEVATED CONCERN", "Weak", "Negative", 2.0)         # -4 qualitative, capped at -2; security +1
    assert r["issuer"]["grade"] == "CR6" and r["facility"]["grade"] == "CR5"
    assert any(s[0] == "Cap on adjustments" for s in r["steps"])
    assert rp.rate(95, None, "Strong", "Stable", 3.0)["facility"]["grade"] == "CR1"   # cannot go above the top grade
    assert rp.rate(np.nan) is None


@pytest.fixture
def blocks():
    vals = pd.Series({"icr": 6.0, "dscr": 0.8, "current_ratio": 1.6, "nd_ebitda": 4.5})
    ctx = dict(name="Test Co", sector="Cement", as_of="data to Mar-2026", vals=vals, flags=[("R", "Operating cash flow is negative.")],
               rating=rp.rate(60, "WATCH"), pillars={"Liquidity": 70.0}, snap=dict(cols=["FY2026"], rows=[("Revenue (Rs cr)", ["1,000"])]),
               analyst=dict(name="A. Analyst", purpose="Working capital", collateral_type="Receivables & inventory",
                            collateral_value=300, management="Weak", outlook="Stable", notes="Promoter pledge ≥ 40% − check."),
               security_cover=0.6, override="Not supportable on current numbers", override_reason="DSCR below 1x",
               req=dict(type="Term loan (amortising)", amount=500, rate=0.1, tenor=5, target=1.25, capacity=120),
               charts=dict(pillars=rp.chart_pillars({"Liquidity": 70.0, "Returns": 40.0}, "Test Co")))
    return rp.build_memo(ctx)


def test_memo_content(blocks):
    md = rp.to_markdown(blocks)
    assert "Not supportable on current numbers" in md and "Analyst override" in md
    assert "Security cover is 0.60x - below 1.0x" in md                     # weak security becomes a condition
    assert "Promoter pledge ≥ 40% − check." in md
    assert "[Chart:" in md                                                  # charts noted in the text version


def test_pdf_and_docx_render(blocks):
    pdf = rp.to_pdf(blocks)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 5000
    from docx import Document
    d = Document(io.BytesIO(rp.to_docx(blocks)))
    text = "\n".join(p.text for p in d.paragraphs)
    assert "Credit appraisal note - Test Co" in text and len(d.inline_shapes) == 1 and len(d.tables) >= 3


def test_relative_standing_flags_sector_bias():
    ntpc_like = rp.relative_standing(sr_score=72, tb_score=55, sector="Power & Utilities")     # weak grade, normal for sector
    assert ntpc_like["bias"] and "whole sector" in ntpc_like["text"]
    laggard = rp.relative_standing(sr_score=39, tb_score=62, sector="Steel & Metals")          # genuinely weak vs peers
    assert not laggard["bias"] and "weaker names" in laggard["text"]
