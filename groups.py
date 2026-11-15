"""Business groups among the dashboard's companies, for group-exposure checks.

Indian credit risk is often group risk: lenders look at the whole family of companies because cash, guarantees and
reputation move between them (IL&FS, ADAG, Videocon). Only companies in the dashboard are listed; membership is by
promoter group, as commonly reported. Central public-sector enterprises are grouped separately because their
support comes from the government, not from sister companies.
"""
GROUPS = {
    "Tata": ["TCS.NS", "TATASTEEL.NS", "TMPV.NS", "TMCV.NS", "TATAPOWER.NS", "TATACONSUM.NS", "TATAELXSI.NS",
             "TATATECH.NS"],
    "Adani": ["ADANIPORTS.NS", "ADANIPOWER.NS", "ADANIGREEN.NS", "ADANIENSOL.NS", "ATGL.NS", "AMBUJACEM.NS", "ACC.NS"],
    "Vedanta (Anil Agarwal)": ["VEDL.NS", "HINDZINC.NS"],
    "JSW (Sajjan Jindal)": ["JSWSTEEL.NS", "JSWENERGY.NS", "JSWINFRA.NS"],
    "Jindal (Naveen / Ratan Jindal)": ["JINDALSTEL.NS", "JSL.NS"],
    "Aditya Birla": ["HINDALCO.NS", "ULTRACEMCO.NS"],
    "Larsen & Toubro": ["LT.NS", "LTM.NS", "LTTS.NS"],
    "Mahindra": ["M&M.NS", "TECHM.NS"],
    "Reliance (Anil Ambani)": ["RPOWER.NS"],
    "Reliance (Mukesh Ambani)": ["RELIANCE.NS"],
    "Torrent": ["TORNTPOWER.NS", "TORNTPHARM.NS"],
    "Zydus": ["ZYDUSLIFE.NS", "ZYDUSWELL.NS"],
    "Hinduja": ["ASHOKLEY.NS"],
    "Murugappa": ["TIINDIA.NS"],
    "Dalmia": ["DALBHARAT.NS"],
    "Central PSUs (Government of India)": ["NTPC.NS", "POWERGRID.NS", "NHPC.NS", "SJVN.NS", "NLCINDIA.NS", "NTPCGREEN.NS",
                                           "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS", "OIL.NS", "MRPL.NS",
                                           "CHENNPETRO.NS", "SAIL.NS", "NMDC.NS", "NATIONALUM.NS", "HINDCOPPER.NS", "MOIL.NS",
                                           "NSLNISP.NS", "RVNL.NS", "IRCON.NS", "NBCC.NS", "ENGINERSIN.NS"],
}
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}


def group_of(ticker: str) -> str | None:
    return GROUP_OF.get(ticker)
