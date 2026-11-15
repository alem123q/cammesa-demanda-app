"""Train and save the model used by the dashboard's forecast page.

    python scripts/train.py

Pre-registered TP2 model: Ridge (alpha = 3.16) on monthly regional demand.
Predictors: region (one-hot), previous month (rezago_1), same month last year (rezago_12), month, quarter.
Lags and target in log(1 + x); predictors standardised.
Train: Jan 2018 - Aug 2019 (180 rows). Test: Sep 2019 - Feb 2020 (54 rows).

models/metadatos.json stores the test metrics against the seasonal rule and the distribution of
percentage errors, which the dashboard uses to show an expected error range.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cammesa.config import DATA_PROCESSED, MODELS, TARGET_IMPROVEMENT  # noqa: E402
from cammesa.features import add_lags, split  # noqa: E402
from cammesa.regression import COLUMNS, LAGS, RIDGE_ALPHA, ridge  # noqa: E402


def errors(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"RMSE_GWh": float(np.sqrt(np.mean(e ** 2)) / 1000), "MAE_GWh": float(np.mean(np.abs(e)) / 1000),
            "MAPE_pct": float(np.mean(np.abs(e) / y) * 100), "sesgo_pct": float(np.mean(e / y) * 100)}


def main() -> None:
    series = pd.read_csv(DATA_PROCESSED / "serie_regional.csv", dtype={"indice_tiempo": str})
    s = add_lags(series)
    train, test = split(s, LAGS)
    print(f"Train: {len(train)} rows ({train['indice_tiempo'].min()} to {train['indice_tiempo'].max()})")
    print(f"Test:  {len(test)} rows ({test['indice_tiempo'].min()} to {test['indice_tiempo'].max()})")

    model = ridge().fit(train[COLUMNS], train["demanda_MWh"])
    y, p_ridge, p_rule = test["demanda_MWh"].to_numpy(), model.predict(test[COLUMNS]), test["rezago_12"].to_numpy()
    m_ridge, m_rule = errors(y, p_ridge), errors(y, p_rule)
    goal = (1 - TARGET_IMPROVEMENT) * m_rule["RMSE_GWh"]
    print(f"Test RMSE: Ridge {m_ridge['RMSE_GWh']:.1f} GWh, rule {m_rule['RMSE_GWh']:.1f} GWh, goal {goal:.1f} GWh")

    err_pct = (y - p_ridge) / p_ridge * 100
    by_region = {r: {"Ridge": errors(g["demanda_MWh"].to_numpy(), g["p"].to_numpy()),
                     "Base": errors(g["demanda_MWh"].to_numpy(), g["rezago_12"].to_numpy())}
                 for r, g in test.assign(p=p_ridge).groupby("region")}
    ranges = {r: {"min_MWh": float(g["demanda_MWh"].min()), "max_MWh": float(g["demanda_MWh"].max())}
              for r, g in s.groupby("region")}
    names = model.regressor_.named_steps["prep"].get_feature_names_out().tolist()
    coef = model.regressor_.named_steps["modelo"].coef_.tolist()

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "ridge_regional.joblib", compress=3)
    meta = {
        "modelo": "Ridge (alpha = 3,16), objetivo y rezagos en log(1 + x)",
        "alpha": RIDGE_ALPHA,
        "entrenamiento": [train["indice_tiempo"].min(), train["indice_tiempo"].max(), int(len(train))],
        "prueba": [test["indice_tiempo"].min(), test["indice_tiempo"].max(), int(len(test))],
        "prueba_ridge": m_ridge, "prueba_base": m_rule, "meta_RMSE_GWh": goal,
        "error_pct_cuantiles": {q: float(np.percentile(err_pct, q)) for q in (5, 10, 50, 90, 95)},
        "por_region": by_region, "rangos": ranges,
        "coeficientes": dict(zip(names, coef)),
        "intercepto": float(model.regressor_.named_steps["modelo"].intercept_),
        "scikit_learn": sklearn.__version__, "fecha_entrenamiento": date.today().isoformat(),
    }
    (MODELS / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", MODELS / "ridge_regional.joblib", "and", MODELS / "metadatos.json")


if __name__ == "__main__":
    main()
