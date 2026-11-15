"""Live inputs: RBI T-bill parsing, OECD fallback, SEBI archive merge, SEBI matching and classification (all offline)."""
import sys, pathlib, types
from datetime import datetime, timedelta, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import macro as mc
import news as nw

RBI_HTML = """<div>Current rates</div><td>91 day T-bills</td><td>: 5.8681%*</td><td>182 day T-bills</td><td>: 6.5183%*</td>
<td>364 day T-bills</td> <td> : 6.6394%* </td>"""
OECD_CSV = ("STRUCTURE,REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE\n"
            "x,IND,M,IR3TIB,2026-09,5.32\nx,IND,M,IR3TIB,2026-10,5.61\nx,IND,M,IR3TIB,2026-08,5.30\n")


def fake_get(responses):
    def get(url, **kw):
        for frag, body in responses.items():
            if frag in url:
                if body is None:
                    raise ConnectionError("down")
                return types.SimpleNamespace(text=body, content=body.encode(), raise_for_status=lambda: None)
        raise ConnectionError("unexpected " + url)
    return get


def test_rbi_then_oecd_then_fallback(monkeypatch):
    monkeypatch.setattr(mc.requests, "get", fake_get({"rbi.org.in": RBI_HTML, "oecd.org": OECD_CSV}))
    r = mc._fetch_rates()
    assert r["rate"] == 6.6394 and r["tbills"] == {"91": 5.8681, "182": 6.5183, "364": 6.6394} and "364-day" in r["source"]
    monkeypatch.setattr(mc.requests, "get", fake_get({"rbi.org.in": None, "oecd.org": OECD_CSV}))
    r = mc._fetch_rates()
    assert r["rate"] == 5.61 and "2026-10" in r["source"] and r["live"]           # latest month, not last row
    monkeypatch.setattr(mc.requests, "get", fake_get({"rbi.org.in": None, "oecd.org": None}))
    r = mc._fetch_rates()
    assert r["rate"] == mc.FALLBACK_RF and not r["live"]


def test_sebi_archive_merge_and_dates():
    now = datetime(2026, 11, 15, tzinfo=timezone.utc)
    old = [{"title": "Order in the matter of ABC Ltd", "url": "u1", "date": (now - timedelta(days=40)).isoformat(), "section": "Order"},
           {"title": "Ancient order", "url": "u0", "date": (now - timedelta(days=2000)).isoformat(), "section": "Order"}]
    new = [{"title": "Order in the matter of ABC Ltd ", "url": "u1", "date": (now - timedelta(days=40)).isoformat(), "section": "Order"},
           {"title": "Settlement Order in respect of XYZ Ltd", "url": "u2", "date": now.isoformat(), "section": "Order"}]
    m = mc.merge_archive(old, new, now)
    assert [x["url"] for x in m] == ["u2", "u1"]                     # deduplicated, newest first, 3-year window
    assert mc._sebi_date("13 Nov, 2026 +0530").day == 13
    assert mc._section("https://www.sebi.gov.in/enforcement/recovery-proceedings/nov-2026/x.html") == "Recovery proceedings"


def test_sebi_matching_and_severity(monkeypatch):
    orders = [{"title": "Adjudication Order in the matter of Wockhardt Ltd", "url": "a", "date": "2026-11-01T00:00:00+05:30", "section": "Order"},
              {"title": "Settlement Order in respect of Wockhardt Ltd", "url": "b", "date": "2026-10-01T00:00:00+05:30", "section": "Order"},
              {"title": "Exemption order in the matter of Wockhardt Ltd", "url": "c", "date": "2026-09-01T00:00:00+05:30", "section": "Order"},
              {"title": "Order in the matter of Arvind Remedies Ltd", "url": "d", "date": "2026-11-01T00:00:00+05:30", "section": "Order"}]
    monkeypatch.setattr(mc, "sebi_orders", lambda: orders)
    found = mc.sebi_for(nw.names_for("Wockhardt", "WOCKPHARMA.NS"), "WOCKPHARMA.NS")
    assert len(found) == 3 and all(x["via"] == "SEBI" for x in found)
    tagged = {x["url"]: x for x in nw.tag(found, ["Wockhardt"], "WOCKPHARMA.NS")}
    assert tagged["a"]["severity"] == "high" and tagged["b"]["severity"] == "medium" and tagged["c"]["tone"] == "neutral"
