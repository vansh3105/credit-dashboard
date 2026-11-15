"""Shadow-rating model: predicts the rating an Indian agency would give, from public data.

Banks and agencies use this when there are too few defaults to fit a default model ("rating replication"): learn
from the ratings experts already assigned, then turn the predicted rating into a default probability using the
agency's published default history.

Labels   ratings.csv - agency ratings with their source (India Ratings' published list of outstanding ratings, and
         rating-action headlines checked by hand). Several agencies for one company are averaged.
Scale    18 notches: AAA = 1, AA+ = 2, AA = 3 ... BBB- = 10 ... D = 18.
Features (chosen on 54 rated non-financial companies; see Method → Model validation)
         size      log market cap - scale and market access, which agencies weigh heavily
         psu       central public-sector enterprise (government support)
         group     member of a large business group (parent support)
         leverage  net debt / EBITDA (capped -3 .. 8)
         coverage  log(1 + interest cover) (cover capped 0 .. 100)
Model    ridge regression on standardised features (alpha 20), prediction clipped to the scale. Validated by
         leave-one-out: each company is predicted by a model fitted on all the others.
PD       1-year default rate of the predicted rating category from CRISIL Ratings' default study
         (Table 1, monthly static pools, fiscals 2014-2024).
Non-financial companies only: there are too few rated lenders to fit their own model.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

NOTCHES = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "C", "D"]
FEATURES = ["size", "psu", "group", "leverage", "coverage"]
FEATURE_LABEL = {"size": "Size (log market cap)", "psu": "Public-sector (PSU)", "group": "Large business group",
                 "leverage": "Net debt / EBITDA", "coverage": "Interest cover (log)"}
ALPHA = 20.0
# CRISIL Ratings default study FY2024, Table 1: average 1-year cumulative default rates, fiscals 2014-2024
CRISIL_PD = {"AAA": 0.0000, "AA": 0.0005, "A": 0.0007, "BBB": 0.0052, "BB": 0.0301, "B": 0.0825, "C": 0.2420}
CRISIL_SRC = "CRISIL Ratings, Annual Default and Ratings Transition Study FY2024, Table 1 (1-year CDR, FY14-24)"


def notch_of(rating: str) -> int | None:
    r = (rating or "").strip().upper()
    return NOTCHES.index(r) + 1 if r in NOTCHES else None


def rating_of(notch: float) -> str:
    return NOTCHES[int(np.clip(round(notch), 1, len(NOTCHES))) - 1]


def category(rating: str) -> str:
    r = rating.rstrip("+-")
    return {"D": "C"}.get(r, r)


def pd_of(rating: str) -> float:
    return CRISIL_PD.get(category(rating), np.nan)


def features(comp, vals: pd.Series, group_name: str | None) -> dict:
    g = group_name or ""
    icr = vals.get("icr")
    return {"size": np.log(comp.mcap_cr) if comp.mcap_cr and comp.mcap_cr == comp.mcap_cr and comp.mcap_cr > 0 else np.nan,
            "psu": float(g.startswith("Central")),
            "group": float(bool(g) and not g.startswith("Central") and "Anil" not in g),
            "leverage": float(np.clip(vals.get("nd_ebitda"), -3, 8)) if not pd.isna(vals.get("nd_ebitda")) else np.nan,
            "coverage": float(np.log1p(np.clip(icr, 0, 100))) if not pd.isna(icr) else np.nan}


def labels(ratings: pd.DataFrame) -> pd.Series:
    """ticker -> average notch over the agencies that rate it."""
    r = ratings.copy()
    r["n"] = r["rating"].map(notch_of)
    return r.dropna(subset=["n"]).groupby("ticker")["n"].mean()


def _fit(X: pd.DataFrame, y: np.ndarray, alpha=ALPHA):
    med = X.median()
    Xf = X.fillna(med)
    mu, sd = Xf.mean(), Xf.std().replace(0, 1).fillna(1)
    Z = ((Xf - mu) / sd).values
    A = np.c_[np.ones(len(y)), Z]
    I = np.eye(A.shape[1]); I[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + alpha * I, A.T @ y)
    return dict(med=med, mu=mu, sd=sd, beta=beta)


def _pred(p, x: pd.Series) -> float:
    z = ((x[FEATURES].astype(float).fillna(p["med"]) - p["mu"]) / p["sd"]).values
    return float(np.clip(p["beta"][0] + z @ p["beta"][1:], 1, len(NOTCHES)))


def fit(X: pd.DataFrame, y: pd.Series, baseline: pd.Series | None = None) -> dict:
    """X: features by ticker; y: notch by ticker. baseline: an existing grade on the notch scale, for comparison.
    Returns the model with leave-one-out validation."""
    X = X.loc[y.index, FEATURES].astype(float)
    yv = y.values.astype(float)
    p = _fit(X, yv)
    loo = np.array([_pred(_fit(X.drop(index=t), np.delete(yv, i)), X.loc[t]) for i, t in enumerate(X.index)])
    err = np.abs(np.round(loo) - yv)
    rank = lambda a: pd.Series(a).rank()
    naive = np.abs(np.round(np.median(yv)) - yv)
    val = {"n": len(yv), "mae": float(err.mean()), "within1": float((err <= 1).mean()), "within2": float((err <= 2).mean()),
           "rank_corr": float(rank(loo).corr(rank(yv))), "naive_mae": float(naive.mean()), "naive_within1": float((naive <= 1).mean())}
    if baseline is not None:
        b = baseline.loc[y.index].values.astype(float)
        e2 = np.abs(b - yv)
        val.update(base_mae=float(e2.mean()), base_within1=float((e2 <= 1).mean()), base_rank_corr=float(rank(b).corr(rank(yv))))
    coef = pd.Series(p["beta"][1:], index=FEATURES)
    return dict(params=p, validation=val, coef=coef, loo=pd.Series(loo, index=X.index), y=y, X=X)


def predict(model: dict, x: dict) -> dict:
    """x: features() of one company. Returns notch, rating, CRISIL category PD and a +/- 1 MAE range."""
    n = _pred(model["params"], pd.Series(x))
    mae = model["validation"]["mae"]
    r = rating_of(n)
    return dict(notch=n, rating=r, pd=pd_of(r), low=rating_of(n + mae), high=rating_of(n - mae),
                missing=[FEATURE_LABEL[k] for k in FEATURES if pd.isna(x.get(k))])


def ratings_hash(ratings: pd.DataFrame) -> str:
    return hashlib.sha1(ratings.to_csv(index=False).encode()).hexdigest()[:12]


# CR grade -> notch, to compare the rule-based grade on the same scale (CR1 AAA, CR2 AA, CR3 A, CR4 BBB, CR5 BB, CR6 B+,
# CR7 B, CR8 CCC ~ C, CR9 CC/C ~ C)
CR_TO_NOTCH = {0: 1, 1: 3, 2: 6, 3: 9, 4: 12, 5: 14, 6: 15, 7: 17, 8: 17}
