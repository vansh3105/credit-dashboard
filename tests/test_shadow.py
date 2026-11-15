"""Shadow-rating model: rating scale, CRISIL default rates, fitting and leave-one-out validation, and the label file."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sectors
import shadow as shd


def test_scale_and_pd():
    assert shd.notch_of("AAA") == 1 and shd.notch_of("bbb-") == 10 and shd.notch_of("IND AA") is None
    assert shd.rating_of(2.4) == "AA+" and shd.rating_of(99) == "D" and shd.rating_of(-3) == "AAA"
    assert shd.category("AA-") == "AA" and shd.category("D") == "C"
    pds = [shd.CRISIL_PD[k] for k in ("AAA", "AA", "A", "BBB", "BB", "B", "C")]
    assert pds == sorted(pds) and shd.pd_of("BBB+") == 0.0052          # CRISIL FY14-24 table, ordinal


def test_fit_recovers_order_and_validates():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({"size": rng.normal(10, 1.5, n), "psu": rng.integers(0, 2, n).astype(float),
                      "group": rng.integers(0, 2, n).astype(float), "leverage": rng.uniform(-2, 6, n),
                      "coverage": np.log1p(rng.uniform(1, 40, n))}, index=[f"C{i}" for i in range(n)])
    y = (12 - 0.9 * (X["size"] - 10) - 1.5 * X["psu"] - X["group"] + 0.6 * X["leverage"] - 1.2 * X["coverage"]
         + rng.normal(0, 0.4, n)).clip(1, 18)
    m = shd.fit(X, y, baseline=pd.Series(9.0, index=X.index))
    v = m["validation"]
    assert v["n"] == n and v["rank_corr"] > 0.8 and v["mae"] < v["naive_mae"]
    assert m["coef"]["size"] < 0 and m["coef"]["leverage"] > 0 and m["coef"]["coverage"] < 0   # bigger, less levered = better
    p = shd.predict(m, {"size": 13.0, "psu": 1.0, "group": 0.0, "leverage": -1.0, "coverage": 3.5})
    assert shd.notch_of(p["rating"]) <= 4 and p["pd"] <= 0.0005 and not p["missing"]
    assert shd.predict(m, {"size": np.nan, "psu": 0, "group": 0, "leverage": 2, "coverage": 2})["missing"]


def test_ratings_file_is_clean():
    r = pd.read_csv(ROOT / "ratings.csv", dtype=str).fillna("")
    known = {t for d in sectors.SECTORS.values() for t in d}
    assert set(r["ticker"]) <= known
    assert r["rating"].map(shd.notch_of).notna().all()
    assert (r["source"] != "").all() and (r["evidence"] != "").all()     # every label says where it came from
    nonfin = r[~r["sector"].isin(sectors.FIN_SECTORS)]["ticker"].nunique()
    assert nonfin >= 40
