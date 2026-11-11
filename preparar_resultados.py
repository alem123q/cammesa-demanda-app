"""
Precalcula los resultados del TP2 y el TP3 que muestra el tablero.
====================================================================
Reproduce, con los hiperparámetros ya elegidos en los TP, los mismos modelos,
la misma partición temporal y la misma validación cruzada (4 bloques de 27 filas):

    TP2 · regresión:      6 modelos + 3 reglas simples  →  resultados/tp2_*.csv
    TP3 · clasificación:  8 modelos + SVM pre-registrada + 4 reglas  →  resultados/tp3_*.csv
    TP3 · agrupamiento:   k-medias con k de 2 a 8  →  resultados/tp3_grupos.csv

La aplicación lee estos archivos; así lo que se ve coincide con los informes y
el tablero responde al instante. Uso:  python preparar_resultados.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, VotingClassifier)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, average_precision_score, cohen_kappa_score, davies_bouldin_score,
                             f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score,
                             silhouette_score, accuracy_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "resultados"
S = 42
CORTE = "2019-09"
CV = TimeSeriesSplit(n_splits=4, test_size=27)


# ======================================================================= datos
def serie_con_rezagos() -> pd.DataFrame:
    s = pd.read_csv(RAIZ / "datos" / "serie_regional.csv", dtype={"indice_tiempo": str})
    s = s.sort_values(["region", "indice_tiempo"]).reset_index(drop=True)
    g = s.groupby("region")["demanda_MWh"]
    s["rezago_1"], s["rezago_12"], s["rezago_13"] = g.shift(1), g.shift(12), g.shift(13)
    s["trimestre"] = (s["mes"] - 1) // 3 + 1
    s["var_previa"] = s["rezago_1"] / s["rezago_13"] - 1                 # ¿el mes anterior subió frente al año anterior?
    s["log_rezago_1"], s["log_rezago_12"] = np.log1p(s["rezago_1"]), np.log1p(s["rezago_12"])
    s["sube"] = (s["demanda_MWh"] > s["rezago_12"]).astype(int)
    return s


def particion(s: pd.DataFrame, columnas: list[str]):
    e = (s[s["indice_tiempo"] < CORTE].dropna(subset=columnas)
         .sort_values(["indice_tiempo", "region"]).reset_index(drop=True))
    t = s[s["indice_tiempo"] >= CORTE].sort_values(["indice_tiempo", "region"]).reset_index(drop=True)
    return e, t


# ======================================================================= TP2 · regresión
REZ, CAL = ["rezago_1", "rezago_12"], ["mes", "trimestre"]
COLS2 = ["region"] + REZ + CAL


def prep_regresion(lineal: bool, normalizar: bool = True) -> ColumnTransformer:
    if lineal:
        rez = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("esc", StandardScaler() if normalizar else "passthrough")])
        cal = StandardScaler() if normalizar else "passthrough"
        reg = OneHotEncoder(drop="first", handle_unknown="ignore")
    else:                                         # árboles: no necesitan escalar
        rez = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        cal = "passthrough"
        reg = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("region", reg, ["region"]), ("rezagos", rez, REZ), ("calendario", cal, CAL)])


def regresor(modelo, lineal: bool) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=Pipeline([("prep", prep_regresion(lineal)), ("modelo", modelo)]),
                                      func=np.log1p, inverse_func=np.expm1)


MODELOS_TP2 = {   # hiperparámetros elegidos en el TP2 (GridSearchCV con la misma validación)
    "Regresión lineal": (LinearRegression(), True),
    "Ridge (pre-registrado)": (Ridge(alpha=3.16228), True),
    "Lasso": (Lasso(alpha=0.01, max_iter=50000), True),
    "Elastic Net": (ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=50000), True),
    "Bosque aleatorio": (RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=6, random_state=S), False),
    "Gradient boosting": (GradientBoostingRegressor(learning_rate=0.03, n_estimators=100, max_depth=3, random_state=S), False),
}


def metricas_reg(y, p) -> dict:
    e = p - y
    return {"RMSE (GWh)": np.sqrt(np.mean(e ** 2)) / 1000, "MAE (GWh)": mean_absolute_error(y, p) / 1000,
            "MAPE (%)": np.mean(np.abs(e) / y) * 100, "Sesgo (%)": np.mean(e / y) * 100,
            "R²": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)}


def tp2(s: pd.DataFrame) -> None:
    e, t = particion(s, REZ)
    bloques = list(CV.split(e))
    pred_val = pd.concat([e.iloc[b][["region", "indice_tiempo", "demanda_MWh"]].assign(bloque=k + 1)
                          for k, (_, b) in enumerate(bloques)], ignore_index=True)
    pred_pru = t[["region", "indice_tiempo", "demanda_MWh"]].copy()
    pred_ent = e[["region", "indice_tiempo", "demanda_MWh"]].copy()
    filas = []

    def registrar(nombre, val_blq, p_ent, p_pru):
        pred_val[nombre] = np.concatenate(val_blq)
        pred_ent[nombre], pred_pru[nombre] = p_ent, p_pru
        rmse_b = [np.sqrt(np.mean((v - e.iloc[b]["demanda_MWh"].to_numpy()) ** 2)) / 1000 for v, (_, b) in zip(val_blq, bloques)]
        m = metricas_reg(t["demanda_MWh"].to_numpy(), p_pru)
        filas.append({"modelo": nombre, "tipo": "regla simple" if nombre.startswith("Regla") else "modelo",
                      "RMSE validación (GWh)": np.mean(rmse_b), "desvío validación": np.std(rmse_b, ddof=1),
                      **{f"RMSE bloque {k + 1}": r for k, r in enumerate(rmse_b)},
                      "RMSE entrenamiento (GWh)": metricas_reg(e["demanda_MWh"].to_numpy(), p_ent)["RMSE (GWh)"], **m})

    for nombre, (mod, lineal) in MODELOS_TP2.items():
        est = regresor(mod, lineal)
        val = [clone(est).fit(e.iloc[a][COLS2], e.iloc[a]["demanda_MWh"]).predict(e.iloc[b][COLS2]) for a, b in bloques]
        final = clone(est).fit(e[COLS2], e["demanda_MWh"])
        registrar(nombre, val, final.predict(e[COLS2]), final.predict(t[COLS2]))
        print(f"  TP2 {nombre:24s} prueba RMSE {filas[-1]['RMSE (GWh)']:6.1f}")

    registrar("Regla: mismo mes del año anterior", [e.iloc[b]["rezago_12"].to_numpy() for _, b in bloques],
              e["rezago_12"].to_numpy(), t["rezago_12"].to_numpy())
    registrar("Regla: mes anterior", [e.iloc[b]["rezago_1"].to_numpy() for _, b in bloques],
              e["rezago_1"].to_numpy(), t["rezago_1"].to_numpy())
    media = lambda ent, nuevo: nuevo["region"].map(ent.groupby("region")["demanda_MWh"].mean()).to_numpy()
    registrar("Regla: media de la región", [media(e.iloc[a], e.iloc[b]) for a, b in bloques], media(e, e), media(e, t))

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / "tp2_metricas.csv", index=False)
    pred_val.to_csv(SALIDA / "tp2_pred_validacion.csv", index=False)
    pred_pru.to_csv(SALIDA / "tp2_pred_prueba.csv", index=False)
    pred_ent.to_csv(SALIDA / "tp2_pred_entrenamiento.csv", index=False)


# ======================================================================= TP3 · clasificación
NUM3 = ["log_rezago_1", "log_rezago_12", "var_previa", "mes", "trimestre"]
COLS3 = ["region"] + NUM3


def flujo(clf, escalar: bool) -> Pipeline:
    prep = ColumnTransformer([("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
                              ("num", StandardScaler() if escalar else "passthrough", NUM3)])
    return Pipeline([("prep", prep), ("clf", clf)])


LOG = LogisticRegression(C=1, class_weight="balanced", max_iter=5000)
ARB = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight=None, random_state=S)
SVM = SVC(C=1, gamma=0.1, class_weight="balanced", probability=True, random_state=S)
MODELOS_TP3 = {   # hiperparámetros elegidos en el TP3
    "Regresión logística": (LOG, True),
    "k vecinos": (KNeighborsClassifier(n_neighbors=15, weights="uniform"), True),
    "Bayes ingenuo": (GaussianNB(var_smoothing=1e-3), True),
    "Árbol de decisión": (ARB, False),
    "SVM (con búsqueda)": (SVM, True),
    "Bosque aleatorio": (RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=6,
                                                class_weight="balanced", random_state=S), False),
    "Gradient boosting": (GradientBoostingClassifier(learning_rate=0.03, n_estimators=300, max_depth=1, random_state=S), False),
    "Votación blanda": (VotingClassifier([("log", LOG), ("arb", ARB), ("svm", SVM)], voting="soft"), True),
    "SVM pre-registrada (por defecto)": (SVC(probability=True, random_state=S), True),
}


def metricas_clf(y, pred, prob=None) -> dict:
    d = {"Exactitud": accuracy_score(y, pred), "Precisión «sube»": precision_score(y, pred, zero_division=0),
         "Exhaustividad «sube»": recall_score(y, pred, zero_division=0), "F1 «sube»": f1_score(y, pred, zero_division=0),
         "F1 macro": f1_score(y, pred, average="macro", zero_division=0), "Kappa": cohen_kappa_score(y, pred)}
    if prob is not None and len(np.unique(y)) > 1:
        d["AUC ROC"], d["Precisión media (PR)"] = roc_auc_score(y, prob), average_precision_score(y, prob)
    return d


def tp3(s: pd.DataFrame) -> None:
    e, t = particion(s, ["rezago_1", "rezago_12", "rezago_13"])
    bloques = list(CV.split(e))
    i_val = np.concatenate([b for _, b in bloques])
    ye, yt, yv = e["sube"].to_numpy(), t["sube"].to_numpy(), e["sube"].to_numpy()[i_val]
    base_cols = ["region", "indice_tiempo", "mes", "sube", "var_previa", "demanda_MWh", "rezago_12"]
    prob_val = e.iloc[i_val][base_cols].reset_index(drop=True).assign(
        bloque=np.concatenate([[k + 1] * len(b) for k, (_, b) in enumerate(bloques)]))
    prob_pru = t[base_cols].copy()
    filas = []

    for nombre, (clf, esc) in MODELOS_TP3.items():
        est = flujo(clf, esc)
        pv = np.concatenate([clone(est).fit(e.iloc[a][COLS3], ye[a]).predict_proba(e.iloc[b][COLS3])[:, 1] for a, b in bloques])
        final = clone(est).fit(e[COLS3], ye)
        pe, pt = final.predict_proba(e[COLS3])[:, 1], final.predict_proba(t[COLS3])[:, 1]
        # La predicción de clase es la de predict (umbral 0,5 en todos salvo SVC, que decide por margen)
        de = final.predict(e[COLS3]); dt = final.predict(t[COLS3])
        dv = np.concatenate([clone(est).fit(e.iloc[a][COLS3], ye[a]).predict(e.iloc[b][COLS3]) for a, b in bloques])
        prob_val[nombre], prob_pru[nombre] = pv, pt
        prob_val[nombre + " · clase"], prob_pru[nombre + " · clase"] = dv, dt
        mv, me, mt = metricas_clf(yv, dv, pv), metricas_clf(ye, de, pe), metricas_clf(yt, dt, pt)
        filas.append({"modelo": nombre, "tipo": "modelo", **{f"{k} · validación": v for k, v in mv.items()},
                      **{f"{k} · entrenamiento": v for k, v in me.items()}, **{f"{k} · prueba": v for k, v in mt.items()}})
        print(f"  TP3 {nombre:34s} F1 val {mv['F1 «sube»']:.2f} prueba {mt['F1 «sube»']:.2f} AUC {mt['AUC ROC']:.2f}")

    reglas = {"Regla: persistencia (repetir el mes anterior)": lambda d: (d["var_previa"] > 0).astype(int).to_numpy(),
              "Regla: siempre «baja»": lambda d: np.zeros(len(d), dtype=int)}
    for nombre, f in reglas.items():
        dv = f(e.iloc[i_val]); de = f(e); dt = f(t)
        prob_val[nombre], prob_pru[nombre] = dv.astype(float), dt.astype(float)
        prob_val[nombre + " · clase"], prob_pru[nombre + " · clase"] = dv, dt
        mv, me, mt = metricas_clf(yv, dv), metricas_clf(ye, de), metricas_clf(yt, dt, dt.astype(float) if "persist" in nombre else None)
        filas.append({"modelo": nombre, "tipo": "regla simple", **{f"{k} · validación": v for k, v in mv.items()},
                      **{f"{k} · entrenamiento": v for k, v in me.items()}, **{f"{k} · prueba": v for k, v in mt.items()}})

    pd.DataFrame(filas).to_csv(SALIDA / "tp3_metricas.csv", index=False)
    prob_val.to_csv(SALIDA / "tp3_prob_validacion.csv", index=False)
    prob_pru.to_csv(SALIDA / "tp3_prob_prueba.csv", index=False)

    # ---------------- agrupamiento (D.1): las mismas predictoras, escaladas, del entrenamiento
    prep = flujo(LogisticRegression(), True).named_steps["prep"].fit(e[COLS3])
    Z = prep.transform(e[COLS3]); Z = Z.toarray() if hasattr(Z, "toarray") else Z
    Zt = prep.transform(t[COLS3]); Zt = Zt.toarray() if hasattr(Zt, "toarray") else Zt
    pca = PCA(n_components=2, random_state=S).fit(Z)
    g = pd.concat([e[["region", "indice_tiempo", "mes", "sube"]].assign(conjunto="entrenamiento"),
                   t[["region", "indice_tiempo", "mes", "sube"]].assign(conjunto="prueba")], ignore_index=True)
    P = pca.transform(np.vstack([Z, Zt])); g["pc1"], g["pc2"] = P[:, 0], P[:, 1]
    curva = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=S).fit(Z)
        g[f"grupo_k{k}"] = np.concatenate([km.labels_, km.predict(Zt)])
        curva.append({"k": k, "silueta": float(silhouette_score(Z, km.labels_)),
                      "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
                      "ari_sube": float(adjusted_rand_score(ye, km.labels_)),
                      "ari_region": float(adjusted_rand_score(e["region"], km.labels_))})
    g.to_csv(SALIDA / "tp3_grupos.csv", index=False)
    (SALIDA / "tp3_grupos_curva.json").write_text(json.dumps(curva, indent=1), encoding="utf-8")
    print("  Agrupamiento: silueta por k", [round(c["silueta"], 3) for c in curva])


if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    s = serie_con_rezagos()
    tp2(s)
    tp3(s)
    print("Listo:", sorted(p.name for p in SALIDA.iterdir()))
