"""TP3: will demand be above the same month last year ("sube")?

Eight classifiers (hyperparameters from the TP3 search) plus the pre-registered default SVM,
against two rules. Also the k-means grouping of region-months used in the dashboard.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, average_precision_score, cohen_kappa_score,
                             davies_bouldin_score, f1_score, precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV, SEED
from .features import split

NUMERIC = ["log_rezago_1", "log_rezago_12", "var_previa", "mes", "trimestre"]
COLUMNS = ["region"] + NUMERIC
PERSISTENCE_RULE = "Regla: persistencia (repetir el mes anterior)"


def make_classifier(clf, scale: bool) -> Pipeline:
    prep = ColumnTransformer([("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
                              ("num", StandardScaler() if scale else "passthrough", NUMERIC)])
    return Pipeline([("prep", prep), ("clf", clf)])


_LOG = LogisticRegression(C=1, class_weight="balanced", max_iter=5000)
_TREE = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight=None, random_state=SEED)
_SVM = SVC(C=1, gamma=0.1, class_weight="balanced", probability=True, random_state=SEED)
MODELS = {  # display name -> (estimator, needs_scaling)
    "Regresión logística": (_LOG, True),
    "k vecinos": (KNeighborsClassifier(n_neighbors=15, weights="uniform"), True),
    "Bayes ingenuo": (GaussianNB(var_smoothing=1e-3), True),
    "Árbol de decisión": (_TREE, False),
    "SVM (con búsqueda)": (_SVM, True),
    "Bosque aleatorio": (RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=6,
                                                class_weight="balanced", random_state=SEED), False),
    "Gradient boosting": (GradientBoostingClassifier(learning_rate=0.03, n_estimators=300, max_depth=1, random_state=SEED), False),
    "Votación blanda": (VotingClassifier([("log", _LOG), ("arb", _TREE), ("svm", _SVM)], voting="soft"), True),
    "SVM pre-registrada (por defecto)": (SVC(probability=True, random_state=SEED), True),
}


def metrics(y, pred, prob=None) -> dict:
    d = {"Exactitud": accuracy_score(y, pred), "Precisión «sube»": precision_score(y, pred, zero_division=0),
         "Exhaustividad «sube»": recall_score(y, pred, zero_division=0), "F1 «sube»": f1_score(y, pred, zero_division=0),
         "F1 macro": f1_score(y, pred, average="macro", zero_division=0), "Kappa": cohen_kappa_score(y, pred)}
    if prob is not None and len(np.unique(y)) > 1:
        d["AUC ROC"], d["Precisión media (PR)"] = roc_auc_score(y, prob), average_precision_score(y, prob)
    return d


def _row(name, kind, m_val, m_train, m_test) -> dict:
    return {"modelo": name, "tipo": kind, **{f"{k} · validación": v for k, v in m_val.items()},
            **{f"{k} · entrenamiento": v for k, v in m_train.items()}, **{f"{k} · prueba": v for k, v in m_test.items()}}


def run(s: pd.DataFrame, verbose: bool = True) -> dict:
    train, test = split(s, ["rezago_1", "rezago_12", "rezago_13"])
    blocks = list(CV.split(train))
    i_val = np.concatenate([b for _, b in blocks])
    y_train, y_test, y_val = train["sube"].to_numpy(), test["sube"].to_numpy(), train["sube"].to_numpy()[i_val]
    keep = ["region", "indice_tiempo", "mes", "sube", "var_previa", "demanda_MWh", "rezago_12"]
    prob_val = train.iloc[i_val][keep].reset_index(drop=True).assign(
        bloque=np.concatenate([[k + 1] * len(b) for k, (_, b) in enumerate(blocks)]))
    prob_test = test[keep].copy()
    rows = []

    for name, (clf, scale) in MODELS.items():
        est = make_classifier(clf, scale)
        pv = np.concatenate([clone(est).fit(train.iloc[a][COLUMNS], y_train[a]).predict_proba(train.iloc[b][COLUMNS])[:, 1]
                             for a, b in blocks])
        final = clone(est).fit(train[COLUMNS], y_train)
        p_tr, p_te = final.predict_proba(train[COLUMNS])[:, 1], final.predict_proba(test[COLUMNS])[:, 1]
        # Class labels come from predict (threshold 0.5 except SVC, which decides by margin)
        c_tr, c_te = final.predict(train[COLUMNS]), final.predict(test[COLUMNS])
        c_val = np.concatenate([clone(est).fit(train.iloc[a][COLUMNS], y_train[a]).predict(train.iloc[b][COLUMNS])
                                for a, b in blocks])
        prob_val[name], prob_test[name] = pv, p_te
        prob_val[name + " · clase"], prob_test[name + " · clase"] = c_val, c_te
        mv, mt = metrics(y_val, c_val, pv), metrics(y_test, c_te, p_te)
        rows.append(_row(name, "modelo", mv, metrics(y_train, c_tr, p_tr), mt))
        if verbose:
            print(f"  TP3 {name:34s} F1 val {mv['F1 «sube»']:.2f} test {mt['F1 «sube»']:.2f} AUC {mt['AUC ROC']:.2f}")

    rules = {PERSISTENCE_RULE: lambda d: (d["var_previa"] > 0).astype(int).to_numpy(),
             "Regla: siempre «baja»": lambda d: np.zeros(len(d), dtype=int)}
    for name, rule in rules.items():
        c_val, c_tr, c_te = rule(train.iloc[i_val]), rule(train), rule(test)
        prob_val[name], prob_test[name] = c_val.astype(float), c_te.astype(float)
        prob_val[name + " · clase"], prob_test[name + " · clase"] = c_val, c_te
        rows.append(_row(name, "regla simple", metrics(y_val, c_val), metrics(y_train, c_tr),
                         metrics(y_test, c_te, c_te.astype(float) if "persist" in name else None)))

    groups, curve = cluster_region_months(train, test)
    return {"tp3_metricas": pd.DataFrame(rows), "tp3_prob_validacion": prob_val, "tp3_prob_prueba": prob_test,
            "tp3_grupos": groups, "tp3_grupos_curva": curve}


def cluster_region_months(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """k-means (k = 2..8) on the TP3 predictors, scaled with the training set; test months are assigned."""
    prep = make_classifier(LogisticRegression(), True).named_steps["prep"].fit(train[COLUMNS])
    dense = lambda m: m.toarray() if hasattr(m, "toarray") else m
    Z, Zt = dense(prep.transform(train[COLUMNS])), dense(prep.transform(test[COLUMNS]))
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    cols = ["region", "indice_tiempo", "mes", "sube"]
    g = pd.concat([train[cols].assign(conjunto="entrenamiento"), test[cols].assign(conjunto="prueba")], ignore_index=True)
    P = pca.transform(np.vstack([Z, Zt]))
    g["pc1"], g["pc2"] = P[:, 0], P[:, 1]
    curve = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        g[f"grupo_k{k}"] = np.concatenate([km.labels_, km.predict(Zt)])
        curve.append({"k": k, "silueta": float(silhouette_score(Z, km.labels_)),
                      "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
                      "ari_sube": float(adjusted_rand_score(train["sube"], km.labels_)),
                      "ari_region": float(adjusted_rand_score(train["region"], km.labels_))})
    return g, curve
