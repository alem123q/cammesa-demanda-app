"""Evaluation on 2022-2025 with weather, activity and tariffs (see docs/preregistro_2022_2025.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .config import DATA_EXTERNAL, DATA_PROCESSED

TRAIN_START, TEST_START, TEST_END = "2018-06", "2022-01", "2025-12"
NORMALS_END = "2021-12"                     # weather normals use 2016-2021 only
CV_EXT = TimeSeriesSplit(n_splits=4, test_size=6 * 9)   # 4 blocks of 6 months x 9 regions
ALPHAS = [0.01, 0.1, 1, 3, 10, 30, 100]
RULE = "Regla: mismo mes del año anterior"


def _shift(month: str, k: int) -> str:
    return str(pd.Period(month, "M") - k)


def build_table() -> pd.DataFrame:
    """One row per region and month with the target and every candidate predictor."""
    s = pd.read_csv(DATA_PROCESSED / "serie_regional_extendida.csv", dtype={"indice_tiempo": str})
    w = pd.read_csv(DATA_EXTERNAL / "weather_monthly_regions.csv", dtype={"indice_tiempo": str})
    e = pd.read_csv(DATA_EXTERNAL / "emae_monthly.csv", dtype={"indice_tiempo": str}).set_index("indice_tiempo")["emae"]
    p = pd.read_csv(DATA_EXTERNAL / "prices_monthly_regions.csv", dtype={"indice_tiempo": str})

    D = s[["region", "indice_tiempo", "mes", "demanda_MWh"]].copy()
    D["trimestre"] = (D["mes"] - 1) // 3 + 1
    S = s.set_index(["region", "indice_tiempo"])["demanda_MWh"]
    W = w.set_index(["region", "indice_tiempo"])[["hdd", "cdd"]]
    P = p.set_index(["region", "indice_tiempo"])["relative_price"]
    w["m"] = w["indice_tiempo"].str[5:]
    normal = w[w["indice_tiempo"] <= NORMALS_END].groupby(["region", "m"])[["hdd", "cdd"]].mean()

    def get(series, r, month):
        return series.get((r, month), np.nan)

    rows = []
    for r, t in zip(D["region"], D["indice_tiempo"]):
        t1, t12, t13 = _shift(t, 1), _shift(t, 12), _shift(t, 13)
        rows.append({
            "rezago_1": get(S, r, t1), "rezago_12": get(S, r, t12), "rezago_13": get(S, r, t13),
            "d_hdd_real": W.loc[(r, t), "hdd"] - W.loc[(r, t12), "hdd"],
            "d_cdd_real": W.loc[(r, t), "cdd"] - W.loc[(r, t12), "cdd"],
            "d_hdd_normal": normal.loc[(r, t[5:]), "hdd"] - W.loc[(r, t12), "hdd"],
            "d_cdd_normal": normal.loc[(r, t[5:]), "cdd"] - W.loc[(r, t12), "cdd"],
            "emae_yoy": np.log(e.get(t, np.nan) / e.get(t12, np.nan)),
            "emae_yoy_l2": np.log(e.get(_shift(t, 2), np.nan) / e.get(_shift(t, 14), np.nan)),
            "tarifa_yoy_l6": np.log(get(P, r, _shift(t, 6)) / get(P, r, _shift(t, 18))),
        })
    D = pd.concat([D.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    D["yoy_previo"] = np.log(D["rezago_1"] / D["rezago_13"])
    D["objetivo"] = np.log(D["demanda_MWh"] / D["rezago_12"])
    D = D[(D["indice_tiempo"] >= TRAIN_START) & (D["indice_tiempo"] <= TEST_END)]
    return D.sort_values(["indice_tiempo", "region"]).reset_index(drop=True)


REGIONS = None


def _weather_by_region(df: pd.DataFrame, cols: list[str], regions: list[str]) -> pd.DataFrame:
    """Each region gets its own weather coefficient (sensitivity differs: heating in Patagonia, cooling in the north)."""
    return pd.DataFrame({f"{c}·{r}": df[c] * (df["region"] == r) for c in cols for r in regions}, index=df.index)


SPECS = {
    "Ridge del TP2": {"kind": "levels"},
    "Modelo de pronóstico": {"kind": "yoy", "plain": ["yoy_previo", "emae_yoy_l2", "tarifa_yoy_l6"],
                             "weather": ["d_hdd_normal", "d_cdd_normal"]},
    "Modelo explicativo": {"kind": "yoy", "plain": ["yoy_previo", "emae_yoy", "tarifa_yoy_l6"],
                           "weather": ["d_hdd_real", "d_cdd_real"]},
}


def design(df: pd.DataFrame, spec: dict, regions: list[str]) -> pd.DataFrame:
    if spec["kind"] == "levels":
        X = pd.get_dummies(df["region"]).reindex(columns=regions, fill_value=0).iloc[:, 1:].astype(float)
        X["log_rezago_1"], X["log_rezago_12"] = np.log1p(df["rezago_1"]), np.log1p(df["rezago_12"])
        X["mes"], X["trimestre"] = df["mes"], df["trimestre"]
        return X
    return pd.concat([df[spec["plain"]], _weather_by_region(df, spec["weather"], regions)], axis=1)


def fit_predict(train: pd.DataFrame, new: pd.DataFrame, spec: dict, alpha: float) -> np.ndarray:
    regions = sorted(train["region"].unique())
    Xa, Xb = design(train, spec, regions), design(new, spec, regions)
    mu, sd = Xa.mean(), Xa.std().replace(0, 1)
    if spec["kind"] == "levels":
        m = Ridge(alpha=alpha).fit((Xa - mu) / sd, np.log1p(train["demanda_MWh"]))
        return np.expm1(m.predict((Xb - mu) / sd))
    m = Ridge(alpha=alpha).fit((Xa - mu) / sd, train["objetivo"])
    return new["rezago_12"].to_numpy() * np.exp(m.predict((Xb - mu) / sd))


def rmse_gwh(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) / 1000)


def split(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = D[D["indice_tiempo"] < TEST_START].reset_index(drop=True)
    test = D[D["indice_tiempo"] >= TEST_START].reset_index(drop=True)
    return train, test


def validate(D: pd.DataFrame) -> pd.DataFrame:
    """Validation RMSE for every model and alpha (rolling origin inside the training period)."""
    train, _ = split(D)
    blocks = list(CV_EXT.split(train))
    rows = [{"modelo": RULE, "alpha": np.nan,
             **{f"bloque {k + 1}": rmse_gwh(train.iloc[b]["demanda_MWh"], train.iloc[b]["rezago_12"])
                for k, (_, b) in enumerate(blocks)}}]
    for name, spec in SPECS.items():
        for a in ALPHAS:
            rows.append({"modelo": name, "alpha": a, **{
                f"bloque {k + 1}": rmse_gwh(train.iloc[b]["demanda_MWh"],
                                            fit_predict(train.iloc[tr], train.iloc[b], spec, a))
                for k, (tr, b) in enumerate(blocks)}})
    v = pd.DataFrame(rows)
    v["RMSE validación (GWh)"] = v.filter(like="bloque").mean(axis=1)
    return v


def chosen_alphas(v: pd.DataFrame) -> dict:
    m = v.dropna(subset=["alpha"])
    return {name: float(g.loc[g["RMSE validación (GWh)"].idxmin(), "alpha"]) for name, g in m.groupby("modelo")}


def test(D: pd.DataFrame, alphas: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, te = split(D)
    pred = te[["region", "indice_tiempo", "demanda_MWh", "rezago_12"]].copy()
    pred[RULE] = te["rezago_12"].to_numpy()
    for name, spec in SPECS.items():
        pred[name] = fit_predict(train, te, spec, alphas[name])
    rows = []
    y = pred["demanda_MWh"].to_numpy()
    for name in [RULE, *SPECS]:
        p = pred[name].to_numpy()
        row = {"modelo": name, "alpha": alphas.get(name, np.nan), "RMSE (GWh)": rmse_gwh(y, p),
               "MAE (GWh)": float(np.mean(np.abs(p - y)) / 1000), "MAPE (%)": float(np.mean(np.abs(p - y) / y) * 100)}
        for yr, g in pred.groupby(pred["indice_tiempo"].str[:4]):
            row[f"RMSE {yr}"] = rmse_gwh(g["demanda_MWh"], g[name])
        rows.append(row)
    res = pd.DataFrame(rows)
    rule = res.loc[res["modelo"] == RULE, "RMSE (GWh)"].iloc[0]
    res["vs regla (%)"] = (res["RMSE (GWh)"] / rule - 1) * 100
    return res, pred


# ------------------------------------------------------------------ classification (docs/preregistro_clasificacion_2022_2025.md)
C_GRID = [0.01, 0.1, 1, 10]
PERSISTENCE = "Regla: persistencia"


def _sube(df: pd.DataFrame) -> np.ndarray:
    return (df["demanda_MWh"] > df["rezago_12"]).astype(int).to_numpy()


def _logistic(train, new, spec, C):
    from sklearn.linear_model import LogisticRegression
    regions = sorted(train["region"].unique())
    Xa, Xb = design(train, spec, regions), design(new, spec, regions)
    mu, sd = Xa.mean(), Xa.std().replace(0, 1)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit((Xa - mu) / sd, _sube(train))
    return m.predict((Xb - mu) / sd)


def _tp3_svm(train, new):
    from sklearn.svm import SVC
    from .classification import COLUMNS, make_classifier
    def prep(df):
        return df.assign(log_rezago_1=np.log1p(df["rezago_1"]), log_rezago_12=np.log1p(df["rezago_12"]),
                         var_previa=df["rezago_1"] / df["rezago_13"] - 1)
    m = make_classifier(SVC(random_state=42), True).fit(prep(train)[COLUMNS], _sube(train))
    return m.predict(prep(new)[COLUMNS])


def classifiers(train, new, alphas, Cs) -> dict:
    """Class predictions (0/1) of every pre-registered candidate for the rows in ``new``."""
    out = {PERSISTENCE: (new["yoy_previo"] > 0).astype(int).to_numpy(),
           "Siempre «sube»": np.ones(len(new), dtype=int)}
    for name, label in [("Modelo de pronóstico", "Clasificador de pronóstico"),
                        ("Modelo explicativo", "Clasificador explicativo")]:
        out[label] = (fit_predict(train, new, SPECS[name], alphas[name]) > new["rezago_12"].to_numpy()).astype(int)
    out["Logística · variables de pronóstico"] = _logistic(train, new, SPECS["Modelo de pronóstico"], Cs["pronóstico"])
    out["Logística · variables explicativas"] = _logistic(train, new, SPECS["Modelo explicativo"], Cs["explicativo"])
    out["SVM del TP3"] = _tp3_svm(train, new)
    return out


def class_metrics(y, p) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    (tn, fp), (fn, tp) = confusion_matrix(y, p, labels=[0, 1])
    return {"F1 «sube»": f1_score(y, p, zero_division=0), "Kappa": cohen_kappa_score(y, p),
            "Exactitud": accuracy_score(y, p), "Exactitud balanceada": balanced_accuracy_score(y, p),
            "VN (baja ok)": tn, "FP": fp, "FN": fn, "VP (sube ok)": tp}


def validate_classification(D: pd.DataFrame, alphas: dict) -> tuple[pd.DataFrame, dict]:
    """Choose C of the two logistic regressions on validation F1 (test untouched)."""
    from sklearn.metrics import f1_score
    train, _ = split(D)
    blocks = list(CV_EXT.split(train))
    rows = []
    for key, name in [("pronóstico", "Modelo de pronóstico"), ("explicativo", "Modelo explicativo")]:
        for C in C_GRID:
            f1 = [f1_score(_sube(train.iloc[b]), _logistic(train.iloc[a], train.iloc[b], SPECS[name], C), zero_division=0)
                  for a, b in blocks]
            rows.append({"logística": key, "C": C, **{f"bloque {k + 1}": v for k, v in enumerate(f1)},
                         "F1 validación": float(np.mean(f1))})
    v = pd.DataFrame(rows)
    Cs = {k: float(g.loc[g["F1 validación"].idxmax(), "C"]) for k, g in v.groupby("logística")}
    return v, Cs


def test_classification(D: pd.DataFrame, alphas: dict, Cs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, te = split(D)
    y = _sube(te)
    preds = classifiers(train, te, alphas, Cs)
    res = pd.DataFrame([{"modelo": k, **class_metrics(y, p)} for k, p in preds.items()])
    base = res.set_index("modelo").loc[PERSISTENCE]
    res["supera a la persistencia (F1 y kappa)"] = (res["F1 «sube»"] > base["F1 «sube»"]) & (res["Kappa"] > base["Kappa"])
    res.loc[res["modelo"] == PERSISTENCE, "supera a la persistencia (F1 y kappa)"] = np.nan
    out = te[["region", "indice_tiempo", "demanda_MWh", "rezago_12"]].assign(sube=y, **preds)
    return res, out
