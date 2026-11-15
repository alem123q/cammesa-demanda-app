"""TP2: forecast next month's demand per region.

Six models (hyperparameters chosen with GridSearchCV on the same validation) against three simple
rules. Ridge was fixed before opening the test set; the others are reported for comparison.
Model names and metric labels are in Spanish because the dashboard displays them as they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CV, SEED
from .features import split

LAGS, CALENDAR = ["rezago_1", "rezago_12"], ["mes", "trimestre"]
COLUMNS = ["region"] + LAGS + CALENDAR
RIDGE_ALPHA = 3.16228          # minimum of the validation curve (pre-registered)
SEASONAL_RULE = "Regla: mismo mes del año anterior"


def preprocessor(linear: bool, scale: bool = True) -> ColumnTransformer:
    """Linear models: log lags, then standardise. Trees: log lags only (scaling does not change them)."""
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    if linear:
        lags = Pipeline([("log", log), ("esc", StandardScaler() if scale else "passthrough")])
        cal = StandardScaler() if scale else "passthrough"
        reg = OneHotEncoder(drop="first", handle_unknown="ignore")
    else:
        lags, cal, reg = log, "passthrough", OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("region", reg, ["region"]), ("rezagos", lags, LAGS), ("calendario", cal, CALENDAR)])


def make_regressor(model, linear: bool, scale: bool = True) -> TransformedTargetRegressor:
    """Pipeline with the target in log(1 + x); predictions come back in MWh."""
    return TransformedTargetRegressor(regressor=Pipeline([("prep", preprocessor(linear, scale)), ("modelo", model)]),
                                      func=np.log1p, inverse_func=np.expm1)


def ridge() -> TransformedTargetRegressor:
    """The pre-registered model."""
    return make_regressor(Ridge(alpha=RIDGE_ALPHA), linear=True)


MODELS = {  # display name -> (estimator, is_linear)
    "Regresión lineal": (LinearRegression(), True),
    "Ridge (pre-registrado)": (Ridge(alpha=RIDGE_ALPHA), True),
    "Lasso": (Lasso(alpha=0.01, max_iter=50000), True),
    "Elastic Net": (ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=50000), True),
    "Bosque aleatorio": (RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=6, random_state=SEED), False),
    "Gradient boosting": (GradientBoostingRegressor(learning_rate=0.03, n_estimators=100, max_depth=3, random_state=SEED), False),
}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Errors in GWh and percentages (inputs in MWh)."""
    e = p - y
    return {"RMSE (GWh)": np.sqrt(np.mean(e ** 2)) / 1000, "MAE (GWh)": mean_absolute_error(y, p) / 1000,
            "MAPE (%)": np.mean(np.abs(e) / y) * 100, "Sesgo (%)": np.mean(e / y) * 100,
            "R²": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)}


def run(s: pd.DataFrame, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Validate and test every model and rule. Returns the metrics table and the predictions."""
    train, test = split(s, LAGS)
    blocks = list(CV.split(train))
    keys = ["region", "indice_tiempo", "demanda_MWh"]
    pred_val = pd.concat([train.iloc[b][keys].assign(bloque=k + 1) for k, (_, b) in enumerate(blocks)], ignore_index=True)
    pred_test, pred_train = test[keys].copy(), train[keys].copy()
    rows = []

    def record(name, val_blocks, p_train, p_test):
        pred_val[name] = np.concatenate(val_blocks)
        pred_train[name], pred_test[name] = p_train, p_test
        rmse_b = [np.sqrt(np.mean((v - train.iloc[b]["demanda_MWh"].to_numpy()) ** 2)) / 1000
                  for v, (_, b) in zip(val_blocks, blocks)]
        rows.append({"modelo": name, "tipo": "regla simple" if name.startswith("Regla") else "modelo",
                     "RMSE validación (GWh)": np.mean(rmse_b), "desvío validación": np.std(rmse_b, ddof=1),
                     **{f"RMSE bloque {k + 1}": r for k, r in enumerate(rmse_b)},
                     "RMSE entrenamiento (GWh)": metrics(train["demanda_MWh"].to_numpy(), p_train)["RMSE (GWh)"],
                     **metrics(test["demanda_MWh"].to_numpy(), p_test)})

    for name, (model, linear) in MODELS.items():
        est = make_regressor(model, linear)
        val = [clone(est).fit(train.iloc[a][COLUMNS], train.iloc[a]["demanda_MWh"]).predict(train.iloc[b][COLUMNS])
               for a, b in blocks]
        final = clone(est).fit(train[COLUMNS], train["demanda_MWh"])
        record(name, val, final.predict(train[COLUMNS]), final.predict(test[COLUMNS]))
        if verbose:
            print(f"  TP2 {name:24s} test RMSE {rows[-1]['RMSE (GWh)']:6.1f} GWh")

    record(SEASONAL_RULE, [train.iloc[b]["rezago_12"].to_numpy() for _, b in blocks],
           train["rezago_12"].to_numpy(), test["rezago_12"].to_numpy())
    record("Regla: mes anterior", [train.iloc[b]["rezago_1"].to_numpy() for _, b in blocks],
           train["rezago_1"].to_numpy(), test["rezago_1"].to_numpy())
    region_mean = lambda fit, new: new["region"].map(fit.groupby("region")["demanda_MWh"].mean()).to_numpy()
    record("Regla: media de la región", [region_mean(train.iloc[a], train.iloc[b]) for a, b in blocks],
           region_mean(train, train), region_mean(train, test))

    return {"tp2_metricas": pd.DataFrame(rows), "tp2_pred_validacion": pred_val,
            "tp2_pred_prueba": pred_test, "tp2_pred_entrenamiento": pred_train}
