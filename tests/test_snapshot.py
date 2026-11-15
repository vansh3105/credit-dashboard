"""Nightly snapshot: lossless JSON round-trip, freshness rules, fallback order, and the what-changed comparison."""
import sys, pathlib
from datetime import datetime, timedelta, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import data as dt
import snapshot as sn


def make_company():
    fin = pd.DataFrame({"rev": [100.0, 120.5], "debt": [np.nan, 40.0]}, index=pd.Index([2025, 2026]))
    return dt.Company("ABC.NS", "ABC Ltd", fin, mcap_cr=1234.5, raw_values={2026: [1.0, 2.5]}, gov={"overall": 3},
                      missing=["sga"], period_end="Mar-2026", fetched_at="15 Nov 2026, 06:00 IST")


def test_round_trip_is_lossless():
    c = make_company()
    q = pd.DataFrame({"rev": [10.0, np.nan]}, index=pd.to_datetime(["2026-06-30", "2026-09-30"]))
    item = {"title": "ABC raises Rs 500 crore", "date": datetime(2026, 11, 1, tzinfo=timezone.utc), "tone": "neutral", "rating": None}
    snap = {"version": sn.VERSION, "built": datetime.now(timezone.utc).isoformat(),
            "companies": {"ABC.NS": sn.company_out(c)}, "quarterly": {"ABC.NS": {"is": sn.df_out(q), "bs": sn.df_out(pd.DataFrame())}},
            "prices": sn.df_out(q.rename(columns={"rev": "ABC.NS"})),
            "news": {"ABC.NS": sn.news_out({"items": [item], "errors": [], "fetched": datetime.now(timezone.utc), "names": ["ABC"]})}}
    back = sn.loads(sn.dumps(snap))
    c2 = sn.company_in(back["companies"]["ABC.NS"])
    pd.testing.assert_frame_equal(c2.fin, c.fin, check_dtype=False, check_index_type=False)
    assert (c2.mcap_cr, c2.raw_values, c2.gov, c2.missing, c2.period_end) == (c.mcap_cr, c.raw_values, c.gov, c.missing, c.period_end)
    q2 = sn.df_in(back["quarterly"]["ABC.NS"]["is"])
    assert list(q2.index) == list(q.index) and np.isnan(q2.iloc[1, 0])
    assert sn.df_in(back["quarterly"]["ABC.NS"]["bs"]).empty
    n2 = sn.news_in(back["news"]["ABC.NS"])
    assert n2["items"][0]["date"] == item["date"]


def test_freshness_and_fallback_order(tmp_path, monkeypatch):
    c = make_company()
    def write(hours_old):
        snap = {"version": sn.VERSION, "built": (datetime.now(timezone.utc) - timedelta(hours=hours_old)).isoformat(),
                "companies": {"ABC.NS": sn.company_out(c)}, "quarterly": {}, "prices": None, "news": {}}
        p = tmp_path / f"s{hours_old}.json.gz"
        p.write_bytes(sn.dumps(snap))
        monkeypatch.setattr(sn, "SNAP_SRC", str(p))
        sn.reset(); dt.clear_cache(); monkeypatch.setattr(dt, "PREFER_LIVE", False)
    live_calls = []
    monkeypatch.setattr(dt, "fetch_live", lambda t, n, retries=2: (live_calls.append(t), (None, "Yahoo down"))[1])

    write(2)                                              # fresh: served from the snapshot, Yahoo never called
    comp, err = dt.get_company("ABC.NS", "ABC Ltd")
    assert comp is not None and err is None and not live_calls and dt.source()["mode"] == "snapshot"

    write(72)                                             # stale: try Yahoo first, fall back to the saved copy
    comp, err = dt.get_company("ABC.NS", "ABC Ltd")
    assert live_calls == ["ABC.NS"] and "saved copy" in comp.fetched_at and dt.source()["mode"] == "live"

    comp, err = dt.get_company("XYZ.NS", "Unknown")      # not in the snapshot and Yahoo down: a clean failure
    assert comp is None and err == "Yahoo down"
    sn.reset()


def test_changes_between_two_nights():
    rows = lambda d, g, sig, news, score: {"date": d, "ticker": "ABC.NS", "name": "ABC", "grade": g, "signal": sig,
                                           "ew_level": "COMFORTABLE", "news": news, "score": score}
    h = pd.DataFrame([rows("2026-11-14", "CR4", "COMFORTABLE", "G", 72.0), rows("2026-11-15", "CR5", "WATCH", "R", 64.0),
                      {**rows("2026-11-15", "CR2", "COMFORTABLE", "G", 85.0), "ticker": "NEW.NS"}])
    h["date"] = pd.to_datetime(h["date"])
    ch = sn.changes(h)
    assert set(ch["What"]) == {"Grade", "Headline signal", "News flow", "Score"}
    assert (ch["Direction"] == "worse").all() and "NEW.NS" not in set(ch["ticker"])
    assert sn.changes(h[h["date"] == "2026-11-15"]).empty      # one night only: nothing to compare
