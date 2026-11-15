"""Live market inputs that are not company data: the risk-free rate and SEBI enforcement orders.

Risk-free rate (for the Merton model, horizon 1 year), first source that answers:
  1. RBI home page - cut-off yield of the latest 364-day Treasury bill auction (also 91- and 182-day)
  2. OECD SDMX API - India short-term interest rate (3-month, monthly average) and 10-year government bond yield
  3. 6.5% fixed fallback
SEBI orders: SEBI's public RSS feed holds only the last few days, so the nightly snapshot keeps every item it has
seen (archive grows from the day the job started). Orders are matched to companies by name in news.get_news().
SEBI's search pages block automated requests, so they are not used.
"""
from __future__ import annotations

import email.utils
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

_UA = {"User-Agent": "Mozilla/5.0 (credit-dashboard)"}
_TTL = 6 * 3600
_LOCK = threading.Lock()
_CACHE: dict = {}
FALLBACK_RF = 6.5
ARCHIVE_DAYS = 3 * 365


def _cached(key, fn):
    with _LOCK:
        hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    val = fn()
    with _LOCK:
        _CACHE[key] = (time.time() if val else time.time() - _TTL + 120, val)
    return val


def clear_cache():
    with _LOCK:
        _CACHE.clear()

# ---------------------------------------------------------------- risk-free rate


def rbi_tbills() -> dict | None:
    """{'91': 5.87, '182': 6.52, '364': 6.64} from the 'Current rates' panel of rbi.org.in, or None."""
    try:
        r = requests.get("https://www.rbi.org.in/", headers=_UA, timeout=15)
        r.raise_for_status()
        txt = re.sub(r"<[^>]+>", " ", r.text)
        txt = re.sub(r"\s+", " ", txt)
        out = {}
        for tenor in ("91", "182", "364"):
            m = re.search(tenor + r"\s*day T-bills?\s*:?\s*([0-9]+\.[0-9]+)\s*%", txt, re.I)
            if m:
                out[tenor] = float(m.group(1))
        return out if "364" in out and 0 < out["364"] < 25 else None
    except Exception:
        return None


def oecd_rate(measure: str) -> tuple[float, str] | None:
    """Latest monthly value of an OECD financial-market series for India: IR3TIB (3-month) or IRLT (10-year)."""
    try:
        url = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
               f"IND.M.{measure}.PA.....?startPeriod={datetime.now().year - 1}-01&format=csvfile")
        r = requests.get(url, headers=_UA, timeout=20)
        r.raise_for_status()
        lines = [ln.split(",") for ln in r.text.strip().splitlines()]
        head = lines[0]
        ti, vi = head.index("TIME_PERIOD"), head.index("OBS_VALUE")
        obs = sorted((row[ti], float(row[vi])) for row in lines[1:] if len(row) > vi and row[vi])
        return (obs[-1][1], obs[-1][0]) if obs else None
    except Exception:
        return None


def _fetch_rates() -> dict:
    tb = rbi_tbills()
    st_ = oecd_rate("IR3TIB")
    lt = oecd_rate("IRLT")
    if tb:
        rate, src = tb["364"], "RBI: cut-off yield, latest 364-day T-bill auction"
    elif st_:
        rate, src = st_[0], f"OECD: India 3-month rate, {st_[1]} average"
    else:
        rate, src = FALLBACK_RF, "fixed fallback (no live source answered)"
    return {"rate": rate, "source": src, "live": bool(tb or st_), "tbills": tb or {},
            "short_3m": st_, "gsec_10y": lt, "fetched": datetime.now(timezone.utc).isoformat()}


def risk_free() -> dict:
    """{'rate': % p.a., 'source', 'live', 'tbills', 'short_3m', 'gsec_10y', 'fetched'}. Nightly snapshot first."""
    import data as dt
    import snapshot as sn
    s = sn.get()
    if s and s.get("macro") and sn.is_fresh(s) and not dt.PREFER_LIVE:
        return s["macro"]
    res = _cached("rf", _fetch_rates)
    if not res["live"] and s and s.get("macro"):          # live sources down: last night's value beats 6.5%
        return {**s["macro"], "source": s["macro"]["source"] + " (saved nightly copy)"}
    return res

# ---------------------------------------------------------------- SEBI orders


def _section(link: str) -> str:
    if "/enforcement/orders/" in link:
        return "Order"
    if "/enforcement/recovery-proceedings/" in link:
        return "Recovery proceedings"
    if "/enforcement/" in link:
        return "Enforcement"
    if "press-releases" in link:
        return "Press release"
    return "Other"


def sebi_rss() -> list[dict]:
    """Items currently in SEBI's RSS feed (only the last few days)."""
    def fetch():
        try:
            r = requests.get("https://www.sebi.gov.in/sebirss.xml", headers=_UA, timeout=15)
            r.raise_for_status()
            out = []
            for it in ET.fromstring(r.content).findall(".//item"):
                title, link = (it.findtext("title") or "").strip(), (it.findtext("link") or "").strip()
                try:
                    d = email.utils.parsedate_to_datetime(it.findtext("pubDate"))
                except (TypeError, ValueError):
                    d = _sebi_date(it.findtext("pubDate"))
                if title and d:
                    out.append({"title": title, "url": link, "date": d.isoformat(), "section": _section(link)})
            return out
        except Exception:
            return []
    return _cached("sebi", fetch)


def _sebi_date(s: str | None):
    """SEBI writes '13 Nov, 2026 +0530'."""
    try:
        return datetime.strptime((s or "").strip(), "%d %b, %Y %z")
    except ValueError:
        return None


def merge_archive(old: list[dict], new: list[dict], now: datetime | None = None) -> list[dict]:
    """Union of two order lists (dates as ISO strings), deduplicated by title, newest first, last 3 years."""
    now = now or datetime.now(timezone.utc)
    seen, out = set(), []
    for it in sorted(old + new, key=lambda x: x["date"], reverse=True):
        k = re.sub(r"\W", "", it["title"].lower())
        if k in seen or (now - datetime.fromisoformat(it["date"])).days > ARCHIVE_DAYS:
            continue
        seen.add(k)
        out.append(it)
    return out


def sebi_orders() -> list[dict]:
    """Every SEBI enforcement item known: the nightly archive plus whatever the RSS feed shows right now."""
    import snapshot as sn
    s = sn.get()
    return merge_archive(list((s or {}).get("sebi") or []), sebi_rss())


def sebi_for(names: list[str], ticker: str) -> list[dict]:
    """SEBI items naming the company, as news items (via='SEBI')."""
    import news as nw
    out = []
    for it in sebi_orders():
        if nw.mentions(it["title"], names) and not (ticker in nw.EXCLUDE and re.search(nw.EXCLUDE[ticker], nw._norm(it["title"]))):
            out.append({"title": it["title"], "source": f"SEBI · {it['section']}", "url": it["url"],
                        "date": datetime.fromisoformat(it["date"]), "via": "SEBI", "section": it["section"]})
    return out
