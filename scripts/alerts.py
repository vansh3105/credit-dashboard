"""Nightly alert for the companies in watchlist.txt.

    python scripts/alerts.py OUT_DIR/history.csv

Always writes a Markdown summary to $GITHUB_STEP_SUMMARY (visible on the workflow run page). Sends an email only when
ALERT_SMTP_HOST, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD and ALERT_TO are set (GitHub secrets); ALERT_SMTP_PORT defaults to
465 (SSL). Nothing is sent when the watchlist is empty.
"""
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

import snapshot as sn


def watchlist() -> list[str]:
    p = ROOT / "watchlist.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


def report(hist: pd.DataFrame, tickers: list[str]) -> tuple[str, int]:
    """Markdown report and the number of changes for the worse."""
    if hist.empty:
        return "No status history yet.", 0
    d1 = hist["date"].max()
    now = hist[hist["date"] == d1].set_index("ticker")
    ch = sn.changes(hist, tickers)
    worse = int((ch["Direction"] == "worse").sum()) if len(ch) else 0
    lines = [f"# Credit watchlist - {pd.Timestamp(d1):%d %b %Y}", ""]
    if len(ch):
        lines += [f"**{len(ch)} change(s) since {ch.attrs['since']:%d %b}, {worse} for the worse.**", "",
                  "| Company | What | Was | Now | |", "|---|---|---|---|---|"]
        lines += [f"| {r.Company} | {r.What} | {r.Was} | {r.Now} | {r.Direction} |" for r in ch.itertuples()]
    else:
        lines.append("No changes since the previous night." if hist["date"].nunique() >= 2 else "First night - changes appear tomorrow.")
    lines += ["", "| Company | Grade | Score | Signal | Early warning | News |", "|---|---|---|---|---|---|"]
    for t in tickers:
        if t in now.index:
            r = now.loc[t]
            lines.append(f"| {r['name']} | {r['grade']} | {r['score']:.0f} | {str(r['signal']).title()} | "
                         f"{'' if pd.isna(r['ew_level']) else str(r['ew_level']).title()} | "
                         f"{ {'R': 'Concern', 'A': 'Watch', 'G': 'Clear'}.get(r['news'], '') } |")
        else:
            lines.append(f"| {t} | not in the dashboard | | | | |")
    return "\n".join(lines), worse


def main():
    hist = pd.read_csv(sys.argv[1], parse_dates=["date"])
    tickers = watchlist()
    if not tickers:
        print("watchlist.txt is empty - no alert.")
        return
    md, worse = report(hist, tickers)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    print(md)
    host, user, pw, to = (os.environ.get(k) for k in ("ALERT_SMTP_HOST", "ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD", "ALERT_TO"))
    if not all((host, user, pw, to)):
        print("Mail settings not configured - summary only.")
        return
    msg = EmailMessage()
    msg["Subject"] = f"Credit watchlist: {worse} change(s) for the worse" if worse else "Credit watchlist: no deterioration"
    msg["From"], msg["To"] = user, to
    msg.set_content(md)
    port = int(os.environ.get("ALERT_SMTP_PORT") or 465)
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as s:
        s.login(user, pw)
        s.send_message(msg)
    print(f"Email sent to {to}.")


if __name__ == "__main__":
    main()
