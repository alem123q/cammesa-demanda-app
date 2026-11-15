"""The refactored pipeline must reproduce the numbers in the course reports exactly."""
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from cammesa import classification, regression  # noqa: E402
from cammesa.data import load_raw, regional_series  # noqa: E402
from cammesa.features import add_lags, split  # noqa: E402


@pytest.fixture(scope="module")
def series():
    return add_lags(regional_series(load_raw()))


@pytest.fixture(scope="module")
def tp2(series):
    return regression.run(series, verbose=False)["tp2_metricas"].set_index("modelo")


@pytest.fixture(scope="module")
def tp3(series):
    return classification.run(series, verbose=False)["tp3_metricas"].set_index("modelo")


def test_regional_series_shape(series):
    assert len(series) == 342                       # 9 regions x 38 months
    assert series["region"].nunique() == 9


def test_split_sizes(series):
    train, test = split(series, regression.LAGS)
    assert (len(train), len(test)) == (180, 54)


def test_tp2_ridge_vs_rule(tp2):
    assert tp2.loc["Ridge (pre-registrado)", "RMSE (GWh)"] == pytest.approx(76.68, abs=0.01)
    assert tp2.loc[regression.SEASONAL_RULE, "RMSE (GWh)"] == pytest.approx(70.96, abs=0.01)
    assert tp2.loc["Ridge (pre-registrado)", "RMSE validación (GWh)"] == pytest.approx(103.7, abs=0.05)


def test_tp3_svm_and_persistence(tp3):
    f1 = "F1 «sube» · prueba"
    assert tp3.loc["SVM pre-registrada (por defecto)", f1] == pytest.approx(0.41, abs=0.005)
    assert tp3.loc["SVM (con búsqueda)", f1] == pytest.approx(0.54, abs=0.005)
    assert tp3.loc[classification.PERSISTENCE_RULE, f1] == pytest.approx(0.83, abs=0.005)
