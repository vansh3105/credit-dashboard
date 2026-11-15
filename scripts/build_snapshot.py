"""Build the nightly snapshot and the daily status history.

    python scripts/build_snapshot.py --prev PREV_DIR --out OUT_DIR

PREV_DIR holds yesterday's snapshot.json.gz / history.csv (may be empty). Companies, quarters, prices or news that
fail today are carried over from yesterday, so one bad night never blanks the app. Writes OUT_DIR/snapshot.json.gz,
OUT_DIR/history.csv and OUT_DIR/README.md. Exit code 1 only if no company could be loaded at all.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

HIST_DAYS = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N companies per sector (testing)")
    args = ap.parse_args()
    prev = Path(args.prev) if args.prev else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Yesterday's files become the fallback for anything that fails today.
    os.environ["CREDIT_SNAPSHOT"] = str(prev / "snapshot.json.gz") if prev else str(out / "_none")
    import snapshot as sn
    sn.SNAP_SRC = os.environ["CREDIT_SNAPSHOT"]
    import data as dt
    import early_warning as ew
    import insights as ins
    import macro as mc
    import metrics as mt
    import news as nw
    import report as rp
    from sectors import SECTORS

    dt.PREFER_LIVE = True                     # always fetch live; the previous snapshot is only a fallback
    mt.set_strictness("Standard")
    mt.set_basis("Textbook")
    t0 = time.time()
    universe = {t: (sec, n) for sec, d in SECTORS.items() for t, n in list(d.items())[: args.limit or None]}
    print(f"{len(universe)} companies in {len(SECTORS)} sectors")

    # ---- annual statements (a few parallel requests; Yahoo rate-limits bursts)
    companies, carried, mcap_carried = {}, [], []
    def fetch(t):
        comp, err = dt.fetch_live(t, universe[t][1], retries=3)
        return t, comp, err
    with ThreadPoolExecutor(4) as ex:
        for t, comp, err in ex.map(fetch, universe):
            if comp is None or comp.fin.empty:
                comp = sn.company(t)
                if comp is not None:
                    carried.append(t)
            elif pd.isna(comp.mcap_cr) or not comp.gov:      # statements fine but market cap / governance missing
                old = sn.company(t)
                if old is not None:
                    if pd.isna(comp.mcap_cr) and not pd.isna(old.mcap_cr):
                        comp.mcap_cr = old.mcap_cr
                        mcap_carried.append(t)
                    if not comp.gov and old.gov:
                        comp.gov = old.gov
            if comp is not None and not comp.fin.empty:
                companies[t] = comp
    no_mcap = [t for t, c in companies.items() if pd.isna(c.mcap_cr)]
    print(f"market cap: {len(mcap_carried)} carried from last night, still missing: {no_mcap}")
    print(f"annual: {len(companies) - len(carried)} live, {len(carried)} carried over, "
          f"{len(universe) - len(companies)} missing ({time.time() - t0:.0f}s)")
    if not companies:
        print("No company data at all - not publishing.")
        sys.exit(1)

    # ---- quarterly, prices, news
    with ThreadPoolExecutor(4) as ex:
        quarterly = dict(zip(companies, ex.map(dt.get_quarterly, companies)))
    import macro as mc_
    px = dt.get_prices(list(companies) + [t for t, *_ in mc_.market_tickers()])   # companies + sector market context
    with ThreadPoolExecutor(6) as ex:
        news = dict(zip(companies, ex.map(lambda t: nw.get_news(companies[t].name, t), companies)))
    print(f"quarterly {sum(len(q['is']) > 0 for q in quarterly.values())}, prices {px.shape[1]} tickers x {len(px)} days, "
          f"news {sum(len(n['items']) > 0 for n in news.values())} ({time.time() - t0:.0f}s)")

    # ---- risk-free rate and the SEBI order archive (RSS keeps only a few days, so keep everything seen)
    prev_snap = sn.get() or {}
    rates = mc._fetch_rates()
    if not rates["live"] and prev_snap.get("macro"):
        rates = {**prev_snap["macro"], "source": prev_snap["macro"]["source"] + " (carried over)"}
    sebi = mc.merge_archive(list(prev_snap.get("sebi") or []), mc.sebi_rss())
    rf = rates["rate"] / 100
    print(f"risk-free {rates['rate']:.2f}% ({rates['source']}); SEBI archive {len(sebi)} items")

    built = datetime.now(timezone.utc)
    snap = {"version": sn.VERSION, "built": built.isoformat(),
            "companies": {t: sn.company_out(c) for t, c in companies.items()},
            "quarterly": {t: {"is": sn.df_out(q["is"]), "bs": sn.df_out(q["bs"])} for t, q in quarterly.items()},
            "prices": sn.df_out(px), "news": {t: sn.news_out(n) for t, n in news.items()},
            "carried_over": carried, "macro": rates, "sebi": sebi}
    blob = sn.dumps(snap)
    (out / "snapshot.json.gz").write_bytes(blob)

    # ---- one status row per company (textbook basis, standard strictness, default weights)
    today = built.astimezone(dt.IST).date().isoformat()
    rows = []
    for sec in SECTORS:
        peers = {t: c for t, c in companies.items() if universe[t][0] == sec}
        sec_px = px[[t for t in peers if t in px.columns]] if len(px) else pd.DataFrame()
        for t, c in peers.items():
            try:
                m = mt.compute(c.fin, c.mcap_cr, 100.0)
                vals = ins.latest(m)
                flags = mt.red_flags(c.fin, m)
                score = mt.overall_score(mt.pillar_scores(vals, c.gov), mt.DEFAULT_WEIGHTS)
                gi = rp.grade_index(score)
                b = ew.bundle(c, peers, sec_px, quarterly.get(t) or {"is": pd.DataFrame(), "bs": pd.DataFrame()}, rf,
                              news[t]["items"])
                mert = b["mert"] or {}
                rows.append({"date": today, "ticker": t, "sector": sec, "name": c.name,
                             "score": None if pd.isna(score) else round(float(score), 1),
                             "grade": rp.GRADES[gi][1] if gi is not None else None,
                             "signal": ins.verdict(vals, flags)["signal"],
                             "ew_level": b["level"] if b["sigs"] else None,
                             "news": nw.news_signal(news[t]["items"])["status"],
                             "dd": None if pd.isna(mert.get("DD", np.nan)) else round(float(mert["DD"]), 2),
                             "pd": None if pd.isna(mert.get("PD", np.nan)) else float(mert["PD"]),
                             "carried": t in carried})
            except Exception as e:
                print(f"  status row failed for {t}: {type(e).__name__}: {e}")
    hist = pd.DataFrame(rows)
    if prev and (prev / "history.csv").exists():
        old = pd.read_csv(prev / "history.csv")
        old = old[old["date"] != today]
        hist = pd.concat([old, hist], ignore_index=True)
    keep = sorted(hist["date"].unique())[-HIST_DAYS:]
    hist = hist[hist["date"].isin(keep)].sort_values(["date", "sector", "ticker"])
    hist.to_csv(out / "history.csv", index=False)

    (out / "README.md").write_text(
        "# Nightly data snapshot\n\nBuilt automatically by `.github/workflows/snapshot.yml`; do not edit by hand.\n\n"
        f"- Built: {built.astimezone(dt.IST):%d %b %Y, %H:%M IST}\n- Companies: {len(companies)} "
        f"({len(carried)} carried over from the previous night)\n- Snapshot size: {len(blob) / 1e6:.1f} MB\n"
        f"- History: {hist['date'].nunique()} days\n")
    print(f"done in {time.time() - t0:.0f}s: snapshot {len(blob) / 1e6:.1f} MB, {len(rows)} status rows, "
          f"{hist['date'].nunique()} history days")


if __name__ == "__main__":
    main()
