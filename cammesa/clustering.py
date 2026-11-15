"""k-means on the agent profiles, with and without ``frac_caidas`` (k = 2..8)."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import PROFILE_COLUMNS

CLUSTER_SEED = 2026   # seed used in the clustering notebook of the course


def cluster_agents(profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add ``grupo_con_k*`` / ``grupo_sin_k*`` labels and return the silhouette / Davies-Bouldin curves.

    "con" uses the six profile columns; "sin" drops ``frac_caidas``, which is redundant with volatility
    and lowers the silhouette.
    """
    out, curves = profiles.copy(), {}
    variants = {"con": PROFILE_COLUMNS, "sin": [c for c in PROFILE_COLUMNS if c != "frac_caidas"]}
    for name, cols in variants.items():
        Z = StandardScaler().fit_transform(profiles[cols])
        rows = []
        for k in range(2, 9):
            m = KMeans(n_clusters=k, n_init=10, random_state=CLUSTER_SEED).fit(Z)
            rows.append({"k": k, "silueta": float(silhouette_score(Z, m.labels_)),
                         "davies_bouldin": float(davies_bouldin_score(Z, m.labels_))})
            out[f"grupo_{name}_k{k}"] = m.labels_
        curves[name] = rows
    return out, curves
