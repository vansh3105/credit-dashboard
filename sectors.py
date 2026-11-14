"""Sector universe. Edit freely: add/remove NSE tickers (Yahoo format: SYMBOL.NS).

Only NON-FINANCIAL sectors are listed on purpose: Beneish M-Score is not valid for
banks/insurers (Session 9-10) and Altman Z / CCC assume an operating business.
The "industry average" is the median/mean of the peers listed here, so keep >= 6
names per sector for a meaningful benchmark.
Every ticker was checked against Yahoo Finance in Nov 2026 (annual statements available). Companies
are fetched in parallel, so a sector of ~25 loads in roughly 10-20 seconds the first time; the app
shows any ticker that fails to load (renames and delistings happen - e.g. LTIMindtree is now LTM.NS).
"""

SECTORS = {
    "IT Services": {
        "TCS.NS": "TCS", "INFY.NS": "Infosys", "WIPRO.NS": "Wipro", "HCLTECH.NS": "HCL Technologies",
        "TECHM.NS": "Tech Mahindra", "LTM.NS": "LTM (LTIMindtree)", "PERSISTENT.NS": "Persistent Systems",
        "COFORGE.NS": "Coforge", "MPHASIS.NS": "Mphasis", "OFSS.NS": "Oracle Financial Services",
        "LTTS.NS": "L&T Technology Services", "KPITTECH.NS": "KPIT Technologies", "TATAELXSI.NS": "Tata Elxsi",
        "TATATECH.NS": "Tata Technologies", "CYIENT.NS": "Cyient", "SONATSOFTW.NS": "Sonata Software", "BSOFT.NS": "Birlasoft",
        "ZENSARTECH.NS": "Zensar Technologies", "MASTEK.NS": "Mastek", "HAPPSTMNDS.NS": "Happiest Minds",
        "INTELLECT.NS": "Intellect Design Arena", "NEWGEN.NS": "Newgen Software", "ECLERX.NS": "eClerx Services",
        "FSL.NS": "Firstsource Solutions", "SASKEN.NS": "Sasken Technologies", "RSYSTEMS.NS": "R Systems International",
        "LATENTVIEW.NS": "LatentView Analytics",
    },
    "Steel & Metals": {
        "TATASTEEL.NS": "Tata Steel", "JSWSTEEL.NS": "JSW Steel", "SAIL.NS": "SAIL", "JINDALSTEL.NS": "Jindal Steel & Power",
        "HINDALCO.NS": "Hindalco", "VEDL.NS": "Vedanta", "NMDC.NS": "NMDC", "NATIONALUM.NS": "NALCO",
        "HINDZINC.NS": "Hindustan Zinc", "JSL.NS": "Jindal Stainless", "APLAPOLLO.NS": "APL Apollo Tubes",
        "HINDCOPPER.NS": "Hindustan Copper", "RATNAMANI.NS": "Ratnamani Metals", "WELCORP.NS": "Welspun Corp",
        "JINDALSAW.NS": "Jindal Saw", "SHYAMMETL.NS": "Shyam Metalics", "LLOYDSME.NS": "Lloyds Metals & Energy",
        "GPIL.NS": "Godawari Power & Ispat", "SARDAEN.NS": "Sarda Energy & Minerals", "MOIL.NS": "MOIL",
        "NSLNISP.NS": "NMDC Steel",
    },
    "Automobiles & Auto Parts": {
        "MARUTI.NS": "Maruti Suzuki", "M&M.NS": "Mahindra & Mahindra", "TMPV.NS": "Tata Motors (Passenger Vehicles)",
        "TMCV.NS": "Tata Motors (Commercial Vehicles)", "HYUNDAI.NS": "Hyundai Motor India", "BAJAJ-AUTO.NS": "Bajaj Auto",
        "EICHERMOT.NS": "Eicher Motors", "HEROMOTOCO.NS": "Hero MotoCorp", "TVSMOTOR.NS": "TVS Motor",
        "ASHOKLEY.NS": "Ashok Leyland", "FORCEMOT.NS": "Force Motors", "ATHERENERG.NS": "Ather Energy",
        "OLECTRA.NS": "Olectra Greentech", "ESCORTS.NS": "Escorts Kubota", "BOSCHLTD.NS": "Bosch",
        "MOTHERSON.NS": "Samvardhana Motherson", "BHARATFORG.NS": "Bharat Forge", "SONACOMS.NS": "Sona BLW Precision",
        "UNOMINDA.NS": "Uno Minda", "TIINDIA.NS": "Tube Investments of India", "ENDURANCE.NS": "Endurance Technologies",
        "SCHAEFFLER.NS": "Schaeffler India", "EXIDEIND.NS": "Exide Industries", "MRF.NS": "MRF",
        "APOLLOTYRE.NS": "Apollo Tyres", "BALKRISIND.NS": "Balkrishna Industries", "CEATLTD.NS": "CEAT",
    },
    "Power & Utilities": {
        "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid", "TATAPOWER.NS": "Tata Power", "ADANIPOWER.NS": "Adani Power",
        "JSWENERGY.NS": "JSW Energy", "NHPC.NS": "NHPC", "TORNTPOWER.NS": "Torrent Power", "CESC.NS": "CESC",
        "ADANIGREEN.NS": "Adani Green Energy", "ADANIENSOL.NS": "Adani Energy Solutions", "NTPCGREEN.NS": "NTPC Green Energy",
        "SJVN.NS": "SJVN", "NLCINDIA.NS": "NLC India", "ACMESOLAR.NS": "ACME Solar", "RPOWER.NS": "Reliance Power",
        "JPPOWER.NS": "Jaiprakash Power Ventures", "RTNPOWER.NS": "RattanIndia Power", "GIPCL.NS": "Gujarat Industries Power",
        "NAVA.NS": "Nava", "PTC.NS": "PTC India",
    },
    "Pharma": {
        "SUNPHARMA.NS": "Sun Pharma", "CIPLA.NS": "Cipla", "DRREDDY.NS": "Dr. Reddy's", "LUPIN.NS": "Lupin",
        "AUROPHARMA.NS": "Aurobindo Pharma", "ZYDUSLIFE.NS": "Zydus Lifesciences", "TORNTPHARM.NS": "Torrent Pharma",
        "ALKEM.NS": "Alkem Labs", "DIVISLAB.NS": "Divi's Laboratories", "MANKIND.NS": "Mankind Pharma",
        "ABBOTINDIA.NS": "Abbott India", "GLENMARK.NS": "Glenmark", "BIOCON.NS": "Biocon", "IPCALAB.NS": "Ipca Labs",
        "LAURUSLABS.NS": "Laurus Labs", "GLAND.NS": "Gland Pharma", "NATCOPHARM.NS": "Natco Pharma",
        "AJANTPHARM.NS": "Ajanta Pharma", "GRANULES.NS": "Granules India", "JBCHEPHARM.NS": "J.B. Chemicals",
        "ERIS.NS": "Eris Lifesciences", "WOCKPHARMA.NS": "Wockhardt", "PFIZER.NS": "Pfizer India", "SANOFI.NS": "Sanofi India",
        "GLAXO.NS": "GSK Pharma India", "ASTRAZEN.NS": "AstraZeneca Pharma India",
    },
    "FMCG": {
        "HINDUNILVR.NS": "Hindustan Unilever", "ITC.NS": "ITC", "NESTLEIND.NS": "Nestle India", "BRITANNIA.NS": "Britannia",
        "DABUR.NS": "Dabur", "GODREJCP.NS": "Godrej Consumer", "MARICO.NS": "Marico", "COLPAL.NS": "Colgate-Palmolive India",
        "TATACONSUM.NS": "Tata Consumer", "VBL.NS": "Varun Beverages", "UNITDSPR.NS": "United Spirits",
        "UBL.NS": "United Breweries", "RADICO.NS": "Radico Khaitan", "PATANJALI.NS": "Patanjali Foods",
        "AWL.NS": "AWL Agri Business", "EMAMILTD.NS": "Emami", "PGHH.NS": "P&G Hygiene & Health",
        "GILLETTE.NS": "Gillette India", "JYOTHYLAB.NS": "Jyothy Labs", "BIKAJI.NS": "Bikaji Foods",
        "ZYDUSWELL.NS": "Zydus Wellness", "HONASA.NS": "Honasa Consumer", "BAJAJCON.NS": "Bajaj Consumer Care",
    },
    "Oil & Gas": {
        "RELIANCE.NS": "Reliance Industries", "ONGC.NS": "ONGC", "IOC.NS": "Indian Oil", "BPCL.NS": "BPCL",
        "HINDPETRO.NS": "HPCL", "GAIL.NS": "GAIL", "OIL.NS": "Oil India", "MRPL.NS": "MRPL", "PETRONET.NS": "Petronet LNG",
        "IGL.NS": "Indraprastha Gas", "MGL.NS": "Mahanagar Gas", "ATGL.NS": "Adani Total Gas",
        "CHENNPETRO.NS": "Chennai Petroleum", "CASTROLIND.NS": "Castrol India", "GULFOILLUB.NS": "Gulf Oil Lubricants",
        "AEGISLOG.NS": "Aegis Logistics", "HINDOILEXP.NS": "Hindustan Oil Exploration", "DEEPINDS.NS": "Deep Industries",
    },
    "Cement": {
        "ULTRACEMCO.NS": "UltraTech Cement", "AMBUJACEM.NS": "Ambuja Cements", "SHREECEM.NS": "Shree Cement", "ACC.NS": "ACC",
        "DALBHARAT.NS": "Dalmia Bharat", "JKCEMENT.NS": "JK Cement", "RAMCOCEM.NS": "Ramco Cements",
        "NUVOCO.NS": "Nuvoco Vistas", "JKLAKSHMI.NS": "JK Lakshmi Cement", "BIRLACORPN.NS": "Birla Corporation",
        "HEIDELBERG.NS": "Heidelberg Cement India", "STARCEMENT.NS": "Star Cement", "INDIACEM.NS": "India Cements",
        "ORIENTCEM.NS": "Orient Cement", "PRSMJOHNSN.NS": "Prism Johnson", "SAGCEM.NS": "Sagar Cements",
        "NCLIND.NS": "NCL Industries", "KCP.NS": "KCP", "MANGLMCEM.NS": "Mangalam Cement",
    },
    "Infra & Construction": {
        "LT.NS": "Larsen & Toubro", "ADANIPORTS.NS": "Adani Ports", "JSWINFRA.NS": "JSW Infrastructure",
        "GPPL.NS": "Gujarat Pipavav Port", "GMRAIRPORT.NS": "GMR Airports", "IRB.NS": "IRB Infrastructure",
        "RVNL.NS": "Rail Vikas Nigam", "IRCON.NS": "Ircon International", "AFCONS.NS": "Afcons Infrastructure",
        "NCC.NS": "NCC", "KEC.NS": "KEC International", "KALPATARU.NS": "Kalpataru Projects", "NBCC.NS": "NBCC",
        "ENGINERSIN.NS": "Engineers India", "GRINFRA.NS": "G R Infraprojects", "PNCINFRA.NS": "PNC Infratech",
        "HGINFRA.NS": "H.G. Infra Engineering", "KNRCON.NS": "KNR Constructions", "ASHOKA.NS": "Ashoka Buildcon",
        "JKIL.NS": "J. Kumar Infraprojects", "PATELENG.NS": "Patel Engineering", "HCC.NS": "Hindustan Construction Co",
    },
}
