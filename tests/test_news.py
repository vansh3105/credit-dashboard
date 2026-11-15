"""News screen: matching, keyword classification, rating extraction, scale mapping, news signal, and the AI call (mocked)."""
import json, sys, pathlib, types
from datetime import datetime, timedelta, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import news as nw
import report as rp

NOW = datetime(2026, 11, 15, tzinfo=timezone.utc)


def item(title, days=10):
    it = {"title": title, "source": "Test", "url": "", "date": NOW - timedelta(days=days), "via": "Google News"}
    it.update(nw.classify(title))
    it["rating"] = nw.extract_rating(title)
    return it


def test_names_and_matching():
    assert nw.names_for("LTM (LTIMindtree)", "LTM.NS")[0] == "LTIMindtree"
    assert nw.names_for("Tata Motors (Passenger Vehicles)", "TMPV.NS") == ["Tata Motors"]
    assert nw.mentions("Dr Reddy's gets USFDA nod", nw.names_for("Dr. Reddy's", "DRREDDY.NS"))
    assert not nw.mentions("CARE Ratings reaffirms Tata Chemicals", ["Tata Steel"])
    assert not nw.mentions("Steel Authority news", ["SAIL"])            # word boundary, not substring


def test_classification():
    c = nw.classify
    assert c("ICRA downgrades Reliance Power to D") == {"category": "Rating action", "tone": "negative", "severity": "high"}
    assert c("India Ratings upgrades NCDs of Vedanta Aluminium to IND AA+")["tone"] == "positive"
    assert c("Reliance Power faces insolvency plea from US Exim Bank")["severity"] == "high"
    # share-price chatter and broker calls are neutral
    for t in ("Wockhardt Ltd Downgraded to Hold Amid Mixed Signals", "Kotak cuts TCS rating to Add",
              "Tata Steel share price target cut by Nomura", "HCLTech shares slump 3% despite in-line results"):
        assert c(t)["category"] == "Market commentary" and c(t)["tone"] == "neutral", t
    # wording traps found in real headlines
    assert c("Tata Steel develops fire-resistant structural steel")["tone"] == "neutral"
    assert c("Adani Ports CEO resigns; Niraj Bansal named successor")["severity"] == "medium"
    assert c("NTPC terminates contract after contractor default")["severity"] != "high"
    assert c("Vedanta plans to cut debt by over Rs 20,000 crore")["tone"] == "positive"
    assert c("HCLTech probe finds no systems breach")["tone"] == "neutral"


def test_rating_extraction_and_scale():
    r = nw.extract_rating("Sunflag Iron and Steel Secures CRISIL AA-/Stable Rating for Bank Facilities")
    assert (r["agency"], r["rating"], r["outlook"]) == ("CRISIL", "AA-", "Stable")
    r = nw.extract_rating("Moody's upgrades Tata Steel to Baa3; outlook stable")
    assert r["rating"] == "Baa3" and r["scale"] == "global" and r["action"] == "Upgrade"
    # a rating about another company in the same headline is not attributed
    t = "Moody's upgrades Reliance Industries rating to Baa1; TCS, Infosys and Tata Steel also see rating actions"
    assert nw.extract_rating(t, ["Tata Steel"]) is None
    assert nw.extract_rating(t, ["Reliance Industries"])["rating"] == "Baa1"
    assert nw.extract_rating("Crisil sees secondary steel margins rising") is None
    assert [nw.grade_index_of(x) for x in ("AAA", "AA+", "A-", "BBB", "BB+", "B+", "B-", "CCC", "D")] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert nw.grade_index_of("Baa3") == 3 and nw.grade_index_of(None) is None
    assert len(rp.GRADES) == 9


def test_news_signal_and_latest_ratings():
    calm = [item("Tata Steel opens new plant", 5), item("Tata Steel Q2 profit rises 16%", 20)]
    assert nw.news_signal(calm, NOW)["status"] == "G"
    assert nw.news_signal([], NOW)["status"] is None
    assert nw.news_signal([item("SEBI orders forensic audit of XYZ", 30)] + calm, NOW)["status"] == "R"
    assert nw.news_signal([item("SEBI orders forensic audit of XYZ", 200)] + calm, NOW)["status"] == "A"
    assert nw.news_signal([item("CRISIL downgrades XYZ to BBB", 150)], NOW)["status"] == "R"
    lr = nw.latest_ratings([item("ICRA reaffirms XYZ at AA; outlook stable", 100), item("ICRA upgrades XYZ to AA+", 20),
                            item("Fitch affirms XYZ at BBB-", 5)])
    assert lr[0]["agency"] == "ICRA" and lr[0]["rating"] == "AA+" and lr[-1]["scale"] == "global"


def test_ai_analyse_with_fake_client():
    sent = [item("ICRA upgrades XYZ to AA+"), item("XYZ CFO resigns")]
    payload = {"items": [{"id": 0, "relevant": True, "category": "Rating action", "tone": "positive", "severity": "medium", "note": "Upgrade."},
                         {"id": 1, "relevant": True, "category": "Governance & management", "tone": "negative", "severity": "high", "note": "Ask why."}],
               "ratings": [{"id": 0, "agency": "ICRA", "rating": "AA+", "outlook": "", "action": "Upgrade"}],
               "news_risk": "Moderate", "digest": "Upgrade [0] but CFO exit [1].", "watch_points": ["Why did the CFO leave?"]}
    calls = []

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(stop_reason="end_turn", model=kw["model"],
                                         content=[types.SimpleNamespace(type="text", text=json.dumps(payload))],
                                         usage=types.SimpleNamespace(input_tokens=500, output_tokens=200))

    client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=Msgs()))
    out = nw.ai_analyse("XYZ", "Cement", sent, client=client)
    assert out["news_risk"] == "Moderate" and out["usage"] == {"in": 500, "out": 200}
    kw = calls[0]
    assert kw["model"] == nw.AI_MODEL and kw["output_config"]["format"]["type"] == "json_schema"
    assert "[0]" in kw["messages"][0]["content"] and "ICRA upgrades XYZ" in kw["messages"][0]["content"]
    m = nw.ai_by_title(sent, out)
    assert m["XYZ CFO resigns"]["severity"] == "high"
    nw.ai_analyse("XYZ", "Cement", sent, client=client)                 # cached: no second call
    assert len(calls) == 1
    assert nw.ai_input([item("XYZ share price target cut"), item("XYZ raises Rs 500 crore via NCDs")])[0]["title"].startswith("XYZ raises")


def test_memo_renders_news_and_agency_check():
    import pandas as pd
    its = [item("XYZ faces insolvency plea from lender ₹500 crore – “report” 🚀", 5), item("ICRA reaffirms XYZ at AA; outlook stable", 30)]
    news = dict(signal=nw.news_signal(its, NOW), top=its[:1],
                ratings=[dict(r, cr=nw.grade_index_of(r["rating"])) for r in nw.latest_ratings(its)])
    ctx = dict(name="XYZ", sector="Cement", as_of="x", vals=pd.Series({"icr": 3.0}), flags=[], rating=rp.rate(58, "WATCH"),
               pillars={"Liquidity": 60.0}, snap=dict(cols=["FY2026"], rows=[("Revenue (Rs cr)", ["1,000"])]), analyst={}, news=news)
    md = rp.to_markdown(rp.build_memo(ctx))
    assert "Agency cross-check" in md and "ICRA AA" in md and "lower" in md and "News flow" in md
    assert rp.to_pdf(rp.build_memo(ctx))[:4] == b"%PDF" and rp.to_docx(rp.build_memo(ctx))[:2] == b"PK"
