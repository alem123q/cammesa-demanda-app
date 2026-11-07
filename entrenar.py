"""
Entrena y serializa el modelo que usa la aplicación.
=====================================================
Reproduce el modelo pre-registrado del TP2 (Decisión 17 del proyecto):

    Ridge con alpha = 3,16 sobre la demanda mensual de cada región.
    Predictoras: región (indicadores), demanda del mes anterior (rezago 1),
    demanda del mismo mes del año anterior (rezago 12), mes y trimestre.
    Rezagos y objetivo en log(1 + x); predictoras escaladas.
    Entrenamiento: enero 2018 – agosto 2019 (180 filas).
    Prueba:        septiembre 2019 – febrero 2020 (54 filas), abierta una sola vez.

Además de guardar el modelo, guarda en modelo/metadatos.json las métricas de la prueba
(contra la base «mismo mes del año anterior») y la distribución de los errores
porcentuales, que la aplicación usa para mostrar un rango de error esperado.

Uso:  python entrenar.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos" / "serie_regional.csv"
SALIDA = RAIZ / "modelo"

ALPHA = 3.16228                       # mínimo de la curva de validación (Decisión 17)
CORTE_PRUEBA = "2019-09"              # primer mes de la prueba
REZAGOS = ["rezago_1", "rezago_12"]
CALENDARIO = ["mes", "trimestre"]
COLUMNAS = ["region"] + REZAGOS + CALENDARIO


def preparar(serie: pd.DataFrame) -> pd.DataFrame:
    """Agrega los rezagos por región y el trimestre."""
    s = serie.sort_values(["region", "indice_tiempo"]).reset_index(drop=True)
    g = s.groupby("region")["demanda_MWh"]
    s["rezago_1"] = g.shift(1)        # mes anterior
    s["rezago_12"] = g.shift(12)      # mismo mes del año anterior (también es la línea de base)
    s["trimestre"] = (s["mes"] - 1) // 3 + 1
    return s


def construir_modelo() -> TransformedTargetRegressor:
    """Todo el preprocesamiento dentro del Pipeline: así la aplicación no tiene que repetirlo."""
    rezagos = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("esc", StandardScaler())])
    prep = ColumnTransformer([
        ("region", OneHotEncoder(drop="first", handle_unknown="ignore"), ["region"]),
        ("rezagos", rezagos, REZAGOS),
        ("calendario", StandardScaler(), CALENDARIO),
    ])
    flujo = Pipeline([("prep", prep), ("modelo", Ridge(alpha=ALPHA))])
    return TransformedTargetRegressor(regressor=flujo, func=np.log1p, inverse_func=np.expm1)


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"RMSE_GWh": float(np.sqrt(np.mean(e ** 2)) / 1000),
            "MAE_GWh": float(np.mean(np.abs(e)) / 1000),
            "MAPE_pct": float(np.mean(np.abs(e) / y) * 100),
            "sesgo_pct": float(np.mean(e / y) * 100)}


def main() -> None:
    serie = preparar(pd.read_csv(DATOS, dtype={"indice_tiempo": str}))
    ent = serie[(serie["indice_tiempo"] < CORTE_PRUEBA)].dropna(subset=REZAGOS)
    pru = serie[serie["indice_tiempo"] >= CORTE_PRUEBA]
    print(f"Entrenamiento: {len(ent)} filas ({ent['indice_tiempo'].min()} → {ent['indice_tiempo'].max()})")
    print(f"Prueba:        {len(pru)} filas ({pru['indice_tiempo'].min()} → {pru['indice_tiempo'].max()})")

    modelo = construir_modelo().fit(ent[COLUMNAS], ent["demanda_MWh"])

    y = pru["demanda_MWh"].to_numpy()
    p_ridge = modelo.predict(pru[COLUMNAS])
    p_base = pru["rezago_12"].to_numpy()
    m_ridge, m_base = metricas(y, p_ridge), metricas(y, p_base)
    print(f"Prueba · Ridge RMSE {m_ridge['RMSE_GWh']:.1f} GWh · base {m_base['RMSE_GWh']:.1f} GWh "
          f"· meta {0.9 * m_base['RMSE_GWh']:.1f} GWh")

    err_pct = (y - p_ridge) / p_ridge * 100          # cuánto se desvía lo real de lo pronosticado
    por_region = {}
    for reg, g in pru.assign(p=p_ridge).groupby("region"):
        por_region[reg] = {"Ridge": metricas(g["demanda_MWh"].to_numpy(), g["p"].to_numpy()),
                           "Base": metricas(g["demanda_MWh"].to_numpy(), g["rezago_12"].to_numpy())}

    # Rango de valores vistos en el entrenamiento, para validar entradas en la aplicación
    rangos = {reg: {"min_MWh": float(g["demanda_MWh"].min()), "max_MWh": float(g["demanda_MWh"].max())}
              for reg, g in serie.groupby("region")}

    # Explicabilidad: coeficientes de Ridge con sus nombres (escala logarítmica, predictoras escaladas)
    nombres = modelo.regressor_.named_steps["prep"].get_feature_names_out().tolist()
    coef = modelo.regressor_.named_steps["modelo"].coef_.tolist()

    SALIDA.mkdir(exist_ok=True)
    joblib.dump(modelo, SALIDA / "ridge_regional.joblib", compress=3)
    meta = {
        "modelo": "Ridge (alpha = 3,16), objetivo y rezagos en log(1 + x)",
        "alpha": ALPHA,
        "entrenamiento": [ent["indice_tiempo"].min(), ent["indice_tiempo"].max(), int(len(ent))],
        "prueba": [pru["indice_tiempo"].min(), pru["indice_tiempo"].max(), int(len(pru))],
        "prueba_ridge": m_ridge,
        "prueba_base": m_base,
        "meta_RMSE_GWh": 0.9 * m_base["RMSE_GWh"],
        "error_pct_cuantiles": {q: float(np.percentile(err_pct, q)) for q in (5, 10, 50, 90, 95)},
        "por_region": por_region,
        "rangos": rangos,
        "coeficientes": dict(zip(nombres, coef)),
        "intercepto": float(modelo.regressor_.named_steps["modelo"].intercept_),
        "scikit_learn": sklearn.__version__,
        "fecha_entrenamiento": date.today().isoformat(),
    }
    (SALIDA / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Guardado:", SALIDA / "ridge_regional.joblib", "y", SALIDA / "metadatos.json")


if __name__ == "__main__":
    main()
