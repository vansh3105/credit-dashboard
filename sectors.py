"""Sector universe. Edit freely: add/remove NSE tickers (Yahoo format: SYMBOL.NS).

Only NON-FINANCIAL sectors are listed on purpose: Beneish M-Score is not valid for
banks/insurers (Session 9-10) and Altman Z / CCC assume an operating business.
The "industry average" is the median/mean of the peers listed here, so keep >= 6
names per sector for a meaningful benchmark.
Tickers were written from memory and NOT verified against live Yahoo data -
the app shows which tickers failed to load.
"""

SECTORS = {
    "IT Services": {
        "TCS.NS": "TCS", "INFY.NS": "Infosys", "WIPRO.NS": "Wipro", "HCLTECH.NS": "HCL Technologies",
        "TECHM.NS": "Tech Mahindra", "LTIM.NS": "LTIMindtree", "PERSISTENT.NS": "Persistent Systems",
        "COFORGE.NS": "Coforge", "MPHASIS.NS": "Mphasis",
    },
    "Steel & Metals": {
        "TATASTEEL.NS": "Tata Steel", "JSWSTEEL.NS": "JSW Steel", "SAIL.NS": "SAIL",
        "JINDALSTEL.NS": "Jindal Steel & Power", "HINDALCO.NS": "Hindalco", "VEDL.NS": "Vedanta",
        "NMDC.NS": "NMDC", "NATIONALUM.NS": "NALCO",
    },
    "Automobiles": {
        "MARUTI.NS": "Maruti Suzuki", "M&M.NS": "Mahindra & Mahindra", "BAJAJ-AUTO.NS": "Bajaj Auto",
        "EICHERMOT.NS": "Eicher Motors", "HEROMOTOCO.NS": "Hero MotoCorp", "TVSMOTOR.NS": "TVS Motor",
        "ASHOKLEY.NS": "Ashok Leyland", "BOSCHLTD.NS": "Bosch",
    },
    "Power & Utilities": {
        "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid", "TATAPOWER.NS": "Tata Power",
        "ADANIPOWER.NS": "Adani Power", "JSWENERGY.NS": "JSW Energy", "NHPC.NS": "NHPC",
        "TORNTPOWER.NS": "Torrent Power", "CESC.NS": "CESC",
    },
    "Pharma": {
        "SUNPHARMA.NS": "Sun Pharma", "CIPLA.NS": "Cipla", "DRREDDY.NS": "Dr. Reddy's",
        "LUPIN.NS": "Lupin", "AUROPHARMA.NS": "Aurobindo Pharma", "ZYDUSLIFE.NS": "Zydus Lifesciences",
        "TORNTPHARM.NS": "Torrent Pharma", "ALKEM.NS": "Alkem Labs",
    },
    "FMCG": {
        "HINDUNILVR.NS": "Hindustan Unilever", "ITC.NS": "ITC", "NESTLEIND.NS": "Nestle India",
        "BRITANNIA.NS": "Britannia", "DABUR.NS": "Dabur", "GODREJCP.NS": "Godrej Consumer",
        "MARICO.NS": "Marico", "COLPAL.NS": "Colgate-Palmolive India",
    },
    "Oil & Gas": {
        "RELIANCE.NS": "Reliance Industries", "ONGC.NS": "ONGC", "IOC.NS": "Indian Oil",
        "BPCL.NS": "BPCL", "HINDPETRO.NS": "HPCL", "GAIL.NS": "GAIL",
        "OIL.NS": "Oil India", "MRPL.NS": "MRPL",
    },
    "Cement": {
        "ULTRACEMCO.NS": "UltraTech Cement", "AMBUJACEM.NS": "Ambuja Cements", "SHREECEM.NS": "Shree Cement",
        "ACC.NS": "ACC", "DALBHARAT.NS": "Dalmia Bharat", "JKCEMENT.NS": "JK Cement",
        "RAMCOCEM.NS": "Ramco Cements", "NUVOCO.NS": "Nuvoco Vistas",
    },
    "Infra & Construction": {
        "LT.NS": "Larsen & Toubro", "ADANIPORTS.NS": "Adani Ports", "IRB.NS": "IRB Infrastructure",
        "NCC.NS": "NCC", "KEC.NS": "KEC International", "KALPATARU.NS": "Kalpataru Projects",
        "NBCC.NS": "NBCC", "GMRAIRPORT.NS": "GMR Airports",
    },
}
