"""Load the raw CAMMESA file and build the two datasets used in the project.

- ``regional_series``: monthly demand per region (9 regions x 38 months).
- ``agent_profiles``: six numbers that summarise each agent's demand history (868 series).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_FILE

FILLER_CATEGORY = "Mercado a Término"   # rows with filler values in the source; excluded from the regional series


def load_raw(path: Path | str = RAW_FILE) -> pd.DataFrame:
    """Read the CAMMESA file (xlsx or csv) with ``indice_tiempo`` as 'YYYY-MM' text."""
    path = Path(path)
    if path.suffix == ".csv":
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"indice_tiempo": str})
    else:
        raw = pd.read_excel(path)
    raw["indice_tiempo"] = raw["indice_tiempo"].astype(str).str[:7]
    return raw


def regional_series(raw: pd.DataFrame) -> pd.DataFrame:
    """Sum demand per region and month, leaving out the filler category."""
    clean = raw[raw["categoria_tarifa"] != FILLER_CATEGORY]
    out = (clean.groupby(["region", "indice_tiempo"])
           .agg(anio=("anio", "first"), mes=("mes", "first"), demanda_MWh=("demanda_MWh", "sum"),
                cantidad_agentes=("agente_nemo", "nunique"))
           .reset_index())
    return out.sort_values(["region", "indice_tiempo"]).reset_index(drop=True)


PROFILE_COLUMNS = ["log_tamano", "estacionalidad", "volatilidad", "tendencia_anual", "frac_caidas", "cv"]


def _profile(g: pd.DataFrame) -> pd.Series | None:
    """Six numbers describing one series; None if it is too short or mostly zeros."""
    s = g.set_index("periodo")["demanda_MWh"]
    if len(s) < 24 or (s <= 0).mean() > 0.1:
        return None
    pos = s[s > 0]
    change = (s / s.shift(1) - 1)[s.shift(1) > 0].clip(-1, 3)
    month = pos.index.month
    summer, winter = pos[np.isin(month, [12, 1, 2])].mean(), pos[np.isin(month, [6, 7, 8])].mean()
    years = np.array([(p - pos.index[0]).n for p in pos.index]) / 12
    return pd.Series({
        "log_tamano": np.log10(pos.mean()),                       # size (log10 of mean MWh/month)
        "estacionalidad": np.log2(summer / winter),               # summer vs winter
        "volatilidad": change.abs().median(),                     # typical month-to-month change
        "tendencia_anual": np.polyfit(years, np.log10(pos.to_numpy()), 1)[0],
        "frac_caidas": (change < -0.15).mean(),                   # share of months with a drop > 15 %
        "cv": pos.std() / pos.mean(),
        "meses": len(s),
    })


def agent_profiles(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per agent and tariff with its profile and its (unused) labels."""
    d = raw.copy()
    d["periodo"] = pd.PeriodIndex(d["indice_tiempo"], freq="M")
    d["serie"] = d["agente_nemo"] + " | " + d["tarifa"]
    d = d.sort_values(["serie", "periodo"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = d.groupby("serie").apply(_profile).dropna()
    labels = d.groupby("serie")[["categoria_demanda", "categoria_tarifa", "tipo_agente", "region"]].first().loc[prof.index]
    names = d.drop_duplicates("agente_nemo").set_index("agente_nemo")["agente_descripcion"]
    prof["agente_nemo"] = [s.split(" | ")[0] for s in prof.index]
    prof["agente"] = prof["agente_nemo"].map(names)
    return prof.join(labels)
