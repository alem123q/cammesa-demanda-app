"""Paths and protocol constants shared by every module.

Column names in the data (``region``, ``indice_tiempo``, ``demanda_MWh``, ``rezago_12``...)
keep the Spanish names of the CAMMESA dataset and of the course reports, so the results
files, the trained model and the dashboard all speak the same schema.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RAW_FILE = DATA_RAW / "demanda-ultimos-anos.xlsx"   # CAMMESA open data, Jan 2017 - Feb 2020

SEED = 42
TEST_START = "2019-09"        # first test month; the test set is Sep 2019 - Feb 2020 (54 region-months)
CV = TimeSeriesSplit(n_splits=4, test_size=27)   # 4 validation blocks of 3 months x 9 regions
TARGET_IMPROVEMENT = 0.10     # pre-registered goal: 10 % less RMSE than the seasonal rule
