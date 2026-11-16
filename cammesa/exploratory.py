"""Exploratory analysis on the course period (2018-2020): what explains what the seasonal rule misses?

Uses the course data (open-data file, test Sep 2019 - Feb 2020) plus weather, EMAE and the regional CPI.
This is exploration on a test set that had already been seen, so its numbers motivate the
pre-registered evaluation on 2022-2025 (cammesa/extended.py) but are not evidence on their own.
Demand is in GWh throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from .config import DATA_EXTERNAL, DATA_PROCESSED, TEST_START

CV_TP = TimeSeriesSplit(n_splits=4, test_size=27)
DROP = ("2018-09", "2019-08")
RECOVERY = ("2019-09", "2020-02")


def _shift(month: str, k: int) -> str:
    return str(pd.Period(month, "M") - k)


def build_table() -> pd.DataFrame:
    s = pd.read_csv(DATA_PROCESSED / "serie_regional.csv", dtype={"indice_tiempo": str})
    S = s.set_index(["region", "indice_tiempo"])["demanda_MWh"] / 1000
    w = pd.read_csv(DATA_EXTERNAL / "weather_monthly_regions.csv", dtype={"indice_tiempo": str})
    W = w.set_index(["region", "indice_tiempo"])[["hdd", "cdd"]]
    w["m"] = w["indice_tiempo"].str[5:]
    normal = w[w["indice_tiempo"] < "2019-01"].groupby(["region", "m"])[["hdd", "cdd"]].mean()   # 2016-2018
    E = pd.read_csv(DATA_EXTERNAL / "emae_monthly.csv", dtype={"indice_tiempo": str}).set_index("indice_tiempo")["emae"]
    p = pd.read_csv(DATA_EXTERNAL / "prices_monthly_regions.csv", dtype={"indice_tiempo": str})
    P = p.set_index(["region", "indice_tiempo"])["relative_price"]

    months = sorted(s["indice_tiempo"].unique())
    rows = []
    for t in months[12:]:
        t1, t12, t13 = _shift(t, 1), _shift(t, 12), _shift(t, 13)
        for r in sorted(s["region"].unique()):
            row = {"fecha": t, "region": r, "y": S[(r, t)], "lag1": S[(r, t1)], "lag12": S[(r, t12)],
                   "lag13": S.get((r, t13), np.nan),
                   "d_hdd": W.loc[(r, t), "hdd"] - W.loc[(r, t12), "hdd"],
                   "d_cdd": W.loc[(r, t), "cdd"] - W.loc[(r, t12), "cdd"],
                   "dn_hdd": normal.loc[(r, t[5:]), "hdd"] - W.loc[(r, t12), "hdd"],
                   "dn_cdd": normal.loc[(r, t[5:]), "cdd"] - W.loc[(r, t12), "cdd"],
                   "emae_yoy": np.log(E[t] / E[t12]),
                   "emae_yoy_l2": np.log(E[_shift(t, 2)] / E[_shift(t, 14)])}
            for L in (0, 3, 6, 9, 12):
                row[f"tarifa_yoy_l{L}"] = np.log(P.get((r, _shift(t, L)), np.nan) / P.get((r, _shift(t, L + 12)), np.nan))
            rows.append(row)
    D = pd.DataFrame(rows)
    D["yoy1"] = np.log(D["lag1"] / D["lag13"])
    D["target"] = np.log(D["y"] / D["lag12"])
    return D


def _features(df, cols, by_region, regions):
    F = df[cols].copy()
    for r in regions:
        for c in by_region:
            F[f"{c}_{r}"] = df[c] * (df["region"] == r)
    return F


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


VARIANTS = {  # name -> (plain predictors, predictors with one coefficient per region)
    "Last month's change only": (["yoy1"], []),
    "+ weather of the month": (["yoy1", "d_hdd", "d_cdd"], []),
    "+ weather of the month, per region": (["yoy1"], ["d_hdd", "d_cdd"]),
    "+ weather per region + EMAE of the month": (["yoy1", "emae_yoy"], ["d_hdd", "d_cdd"]),
    "+ weather per region + EMAE as published (t-2)": (["yoy1", "emae_yoy_l2"], ["d_hdd", "d_cdd"]),
    "+ normal weather (known in advance)": (["yoy1", "dn_hdd", "dn_cdd"], []),
    "+ normal weather + EMAE as published (t-2)": (["yoy1", "dn_hdd", "dn_cdd", "emae_yoy_l2"], []),
}


def yoy_experiments(alpha: float = 1.0) -> pd.DataFrame:
    """Validation and test RMSE (GWh) of year-over-year models on the course split."""
    D = build_table().dropna(subset=["yoy1"])
    tr = D[D["fecha"] < TEST_START].sort_values(["fecha", "region"]).reset_index(drop=True)
    te = D[D["fecha"] >= TEST_START]
    regions = sorted(D["region"].unique())
    blocks = list(CV_TP.split(tr))

    def fit_pred(a, b, cols, by_region):
        Fa, Fb = _features(a, cols, by_region, regions), _features(b, cols, by_region, regions)
        mu, sd = Fa.mean(), Fa.std().replace(0, 1)
        m = Ridge(alpha=alpha).fit((Fa - mu) / sd, a["target"])
        return b["lag12"] * np.exp(m.predict((Fb - mu) / sd))

    rows = [{"model": "Seasonal rule (change = 0)",
             "validation RMSE": np.mean([_rmse(tr.y.iloc[b], tr.lag12.iloc[b]) for _, b in blocks]),
             "test RMSE": _rmse(te.y, te.lag12), "known in advance": True}]
    for name, (cols, by_region) in VARIANTS.items():
        v = np.mean([_rmse(tr.y.iloc[b], fit_pred(tr.iloc[a], tr.iloc[b], cols, by_region)) for a, b in blocks])
        rows.append({"model": name, "validation RMSE": v, "test RMSE": _rmse(te.y, fit_pred(tr, te, cols, by_region)),
                     "known in advance": not any(c in ("d_hdd", "d_cdd", "emae_yoy") for c in cols + by_region)})
    return pd.DataFrame(rows)


def decomposition(tariff_lag: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Split the year-over-year change of the drop and the recovery into weather, activity (and tariffs).

    A linear model of log(demand / same month last year) on the degree-day differences, the EMAE change and,
    optionally, the real tariff change lagged ``tariff_lag`` months, fitted on every available region-month.
    Contributions are weighted by each region's size. Descriptive, not predictive.
    """
    D = build_table()
    cols = ["d_hdd", "d_cdd", "emae_yoy"] + ([f"tarifa_yoy_l{tariff_lag}"] if tariff_lag is not None else [])
    D = D.dropna(subset=cols)
    m = LinearRegression().fit(D[cols], D["target"])
    labels = {"d_hdd": "heating (cold)", "d_cdd": "cooling (heat)", "emae_yoy": "activity (EMAE)"}
    rows = []
    for name, (a, b) in {"Drop (Sep 2018 - Aug 2019)": DROP, "Recovery (Sep 2019 - Feb 2020)": RECOVERY}.items():
        Q = D[(D["fecha"] >= a) & (D["fecha"] <= b)]
        w = Q["lag12"] / Q["lag12"].sum()
        actual = (w * Q["target"]).sum()
        row = {"period": name, "actual change": 100 * actual}
        for i, c in enumerate(cols):
            row[labels.get(c, "real tariff")] = 100 * (w * Q[c] * m.coef_[i]).sum()
        row["rest"] = row["actual change"] - sum(row[labels.get(c, "real tariff")] for c in cols) - 100 * m.intercept_
        rows.append(row)
    coefs = {labels.get(c, "real tariff"): float(v) for c, v in zip(cols, m.coef_)}
    coefs["R2"] = float(m.score(D[cols], D["target"]))
    return pd.DataFrame(rows), coefs


def tariff_lags() -> pd.DataFrame:
    """Does adding the real tariff change (at several lags) improve the weather + EMAE model?"""
    D = build_table()
    base = ["d_hdd", "d_cdd", "emae_yoy"]
    rows = []
    for L in (0, 3, 6, 9, 12):
        c = f"tarifa_yoy_l{L}"
        E = D.dropna(subset=[c])
        m, m0 = LinearRegression().fit(E[base + [c]], E["target"]), LinearRegression().fit(E[base], E["target"])
        rows.append({"lag (months)": L, "region-months": len(E), "from": E["fecha"].min(),
                     "tariff coefficient": m.coef_[-1], "R2 without": m0.score(E[base], E["target"]),
                     "R2 with": m.score(E[base + [c]], E["target"])})
    return pd.DataFrame(rows)
