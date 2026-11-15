"""Nightly data snapshot: every company's statements, quarterly data, prices and news in one JSON file.

A GitHub Actions job (.github/workflows/snapshot.yml) rebuilds it every night and publishes it on the repo's `data`
branch together with history.csv (one status row per company per day). The app reads it first, so it
  * opens in seconds instead of waiting on 20+ Yahoo requests,
  * keeps working when Yahoo is slow or rate-limiting (the saved copy is used as a fallback), and
  * can show what changed since the previous day.

JSON only (no pickle), so loading a snapshot can never run code.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

REPO_RAW = "https://raw.githubusercontent.com/vansh3105/credit-dashboard/data"
SNAP_SRC = os.environ.get("CREDIT_SNAPSHOT", f"{REPO_RAW}/snapshot.json.gz")   # URL or local path
HIST_SRC = os.environ.get("CREDIT_HISTORY", f"{REPO_RAW}/history.csv")
FRESH_HOURS = 30          # a snapshot younger than this is used directly; older ones only as a fallback
RECHECK = 3600            # look for a newer snapshot at most once an hour
VERSION = 1

_LOCK = threading.Lock()
_STATE = {"t": 0.0, "snap": None, "hist_t": 0.0, "hist": None}

# ---------------------------------------------------------------- (de)serialisation


def df_out(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    dt_index = isinstance(df.index, pd.DatetimeIndex)
    return {"index": [i.isoformat() for i in df.index] if dt_index else [int(i) if isinstance(i, (int, np.integer)) else i for i in df.index],
            "dt": dt_index, "columns": [str(c) for c in df.columns],
            "data": [[None if pd.isna(v) else float(v) for v in row] for row in df.to_numpy(dtype=float)]}


def df_in(d: dict | None) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    idx = pd.to_datetime(d["index"]) if d["dt"] else pd.Index(d["index"])
    if d["dt"] and getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return pd.DataFrame(np.array(d["data"], dtype=float).reshape(len(d["index"]), len(d["columns"])),
                        index=idx, columns=d["columns"])


def company_out(c) -> dict:
    return {"ticker": c.ticker, "name": c.name, "fin": df_out(c.fin), "mcap_cr": None if pd.isna(c.mcap_cr) else float(c.mcap_cr),
            "raw_values": {str(k): [float(x) for x in v] for k, v in (c.raw_values or {}).items()}, "gov": c.gov or {},
            "currency": c.currency, "missing": list(c.missing or []), "period_end": c.period_end, "fetched_at": c.fetched_at}


def company_in(d: dict):
    import data as dt
    return dt.Company(ticker=d["ticker"], name=d["name"], fin=df_in(d["fin"]),
                      mcap_cr=np.nan if d.get("mcap_cr") is None else d["mcap_cr"],
                      raw_values={int(k): v for k, v in (d.get("raw_values") or {}).items()}, gov=d.get("gov") or {},
                      currency=d.get("currency", "INR"), missing=d.get("missing") or [], period_end=d.get("period_end", ""),
                      fetched_at=d.get("fetched_at", ""))


def news_out(res: dict) -> dict:
    return {"items": [{**it, "date": it["date"].isoformat()} for it in res.get("items", [])],
            "errors": res.get("errors", []), "fetched": res["fetched"].isoformat() if res.get("fetched") else None,
            "names": res.get("names", [])}


def news_in(d: dict) -> dict:
    return {"items": [{**it, "date": datetime.fromisoformat(it["date"])} for it in d.get("items", [])],
            "errors": d.get("errors", []), "names": d.get("names", []),
            "fetched": datetime.fromisoformat(d["fetched"]) if d.get("fetched") else None}


def dumps(snap: dict) -> bytes:
    return gzip.compress(json.dumps(snap, separators=(",", ":")).encode(), compresslevel=9)


def loads(blob: bytes) -> dict:
    return json.loads(gzip.decompress(blob))

# ---------------------------------------------------------------- reading


def _read(src: str, binary=True):
    if src.startswith("http"):
        r = requests.get(src, timeout=20)
        r.raise_for_status()
        return r.content if binary else r.text
    with open(src, "rb" if binary else "r") as f:
        return f.read()


def get() -> dict | None:
    """The latest snapshot (parsed lazily per section), or None if there is none. Re-checked at most hourly."""
    with _LOCK:
        if time.time() - _STATE["t"] < RECHECK:
            return _STATE["snap"]
        _STATE["t"] = time.time()
    try:
        raw = loads(_read(SNAP_SRC))
        snap = raw if raw.get("version") == VERSION else None
    except Exception:
        snap = None
    with _LOCK:
        if snap is not None or _STATE["snap"] is None:
            _STATE["snap"] = snap
        return _STATE["snap"]


def built_at(snap: dict | None) -> datetime | None:
    return datetime.fromisoformat(snap["built"]) if snap and snap.get("built") else None


def age_hours(snap: dict | None) -> float:
    b = built_at(snap)
    return (datetime.now(timezone.utc) - b).total_seconds() / 3600 if b else float("inf")


def is_fresh(snap: dict | None) -> bool:
    return age_hours(snap) < FRESH_HOURS


def company(ticker: str):
    s = get()
    d = (s or {}).get("companies", {}).get(ticker)
    return company_in(d) if d else None


def quarterly(ticker: str) -> dict | None:
    s = get()
    d = (s or {}).get("quarterly", {}).get(ticker)
    return {"is": df_in(d.get("is")), "bs": df_in(d.get("bs"))} if d else None


def prices(tickers) -> pd.DataFrame:
    s = get()
    if not s or not s.get("prices"):
        return pd.DataFrame()
    with _LOCK:
        px = _STATE.get("px")
        if px is None or _STATE.get("px_built") != s.get("built"):
            px = df_in(s["prices"])
            _STATE["px"], _STATE["px_built"] = px, s.get("built")
    cols = [t for t in tickers if t in px.columns]
    return px[cols].dropna(how="all") if cols else pd.DataFrame()


def news(ticker: str) -> dict | None:
    s = get()
    d = (s or {}).get("news", {}).get(ticker)
    return news_in(d) if d else None


def history() -> pd.DataFrame:
    """Daily status rows (date, ticker, sector, name, score, grade, signal, ew_level, news, dd, pd). Empty if none yet."""
    with _LOCK:
        if time.time() - _STATE["hist_t"] < RECHECK and _STATE["hist"] is not None:
            return _STATE["hist"]
        _STATE["hist_t"] = time.time()
    try:
        h = pd.read_csv(io.StringIO(_read(HIST_SRC, binary=False)), parse_dates=["date"])
    except Exception:
        h = pd.DataFrame()
    with _LOCK:
        _STATE["hist"] = h
    return h


def reset():
    with _LOCK:
        _STATE.update(t=0.0, snap=None, hist_t=0.0, hist=None, px=None, px_built=None)

# ---------------------------------------------------------------- what changed


SIG_RANK = {"COMFORTABLE": 0, "WATCH": 1, "ELEVATED CONCERN": 2}
NEWS_RANK = {"G": 0, "A": 1, "R": 2}


def changes(hist: pd.DataFrame, tickers=None) -> pd.DataFrame:
    """Rows that moved between the last two snapshot dates: grade, headline signal, early-warning level or news status."""
    if hist is None or hist.empty or hist["date"].nunique() < 2:
        return pd.DataFrame()
    d1, d0 = sorted(hist["date"].unique())[-1], sorted(hist["date"].unique())[-2]
    h = hist if tickers is None else hist[hist["ticker"].isin(list(tickers))]
    a = h[h["date"] == d0].set_index("ticker")
    b = h[h["date"] == d1].set_index("ticker")
    out = []
    for t in b.index.intersection(a.index):
        o, n = a.loc[t], b.loc[t]
        moves = []
        for col, label in (("grade", "Grade"), ("signal", "Headline signal"), ("ew_level", "Early warning"), ("news", "News flow")):
            ov, nv = o.get(col), n.get(col)
            if pd.notna(ov) and pd.notna(nv) and ov != nv:
                moves.append((label, ov, nv))
        if pd.notna(o.get("score")) and pd.notna(n.get("score")) and abs(n["score"] - o["score"]) >= 5:
            moves.append(("Score", f"{o['score']:.0f}", f"{n['score']:.0f}"))
        for label, ov, nv in moves:
            worse = _worse(label, ov, nv)
            out.append({"Company": n.get("name", t), "ticker": t, "What": label, "Was": ov, "Now": nv,
                        "Direction": "worse" if worse else ("better" if worse is False else "changed")})
    df = pd.DataFrame(out, columns=["Company", "ticker", "What", "Was", "Now", "Direction"])
    df.attrs.update(since=pd.Timestamp(d0), until=pd.Timestamp(d1))
    return df


def _worse(label, ov, nv):
    try:
        if label == "Grade":
            return int(str(nv)[2:]) > int(str(ov)[2:])
        if label in ("Headline signal", "Early warning"):
            return SIG_RANK[nv] > SIG_RANK[ov]
        if label == "News flow":
            return NEWS_RANK[nv] > NEWS_RANK[ov]
        if label == "Score":
            return float(nv) < float(ov)
    except (KeyError, ValueError):
        return None
    return None
