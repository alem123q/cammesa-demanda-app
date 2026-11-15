"""Lags, calendar variables and the temporal train/test split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TEST_START


def add_lags(series: pd.DataFrame) -> pd.DataFrame:
    """Add per-region lags and derived variables to the regional series.

    rezago_1   demand of the previous month
    rezago_12  demand of the same month last year (also the seasonal rule)
    rezago_13  demand 13 months earlier (to compute last month's year-over-year change)
    var_previa year-over-year change of the previous month (TP3)
    sube       1 if demand is above the same month last year (TP3 target)
    """
    s = series.sort_values(["region", "indice_tiempo"]).reset_index(drop=True)
    g = s.groupby("region")["demanda_MWh"]
    s["rezago_1"], s["rezago_12"], s["rezago_13"] = g.shift(1), g.shift(12), g.shift(13)
    s["trimestre"] = (s["mes"] - 1) // 3 + 1
    s["var_previa"] = s["rezago_1"] / s["rezago_13"] - 1
    s["log_rezago_1"], s["log_rezago_12"] = np.log1p(s["rezago_1"]), np.log1p(s["rezago_12"])
    s["sube"] = (s["demanda_MWh"] > s["rezago_12"]).astype(int)
    return s


def split(s: pd.DataFrame, required: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (before TEST_START, rows with all ``required`` columns) and test, ordered by month then region.

    The ordering matters: TimeSeriesSplit cuts by row position, so each validation block is 3 whole months.
    """
    train = (s[s["indice_tiempo"] < TEST_START].dropna(subset=required)
             .sort_values(["indice_tiempo", "region"]).reset_index(drop=True))
    test = s[s["indice_tiempo"] >= TEST_START].sort_values(["indice_tiempo", "region"]).reset_index(drop=True)
    return train, test
