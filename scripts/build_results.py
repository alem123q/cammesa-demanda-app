"""Rebuild every dataset and result from the raw CAMMESA file.

    python scripts/build_results.py

raw xlsx -> data/processed/  (regional series, agent profiles and clusters)
         -> results/         (TP2 and TP3 metrics, predictions and region-month clusters)

The dashboard reads these files, so what it shows matches the reports and loads instantly.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from cammesa import classification, regression  # noqa: E402
from cammesa.clustering import cluster_agents  # noqa: E402
from cammesa.config import DATA_PROCESSED, RESULTS  # noqa: E402
from cammesa.data import agent_profiles, load_raw, regional_series  # noqa: E402
from cammesa.features import add_lags  # noqa: E402


def save(outputs: dict, folder: Path) -> None:
    for name, obj in outputs.items():
        if isinstance(obj, (list, dict)):
            (folder / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
        else:
            obj.to_csv(folder / f"{name}.csv", index=False)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Loading raw data")
    raw = load_raw()
    series = regional_series(raw)
    series.to_csv(DATA_PROCESSED / "serie_regional.csv", index=False)
    print(f"  regional series: {len(series)} rows ({series['indice_tiempo'].min()} to {series['indice_tiempo'].max()})")

    print("Agent profiles and clusters")
    profiles, curves = cluster_agents(agent_profiles(raw))
    profiles.reset_index().to_csv(DATA_PROCESSED / "agentes_perfiles.csv", index=False)
    (DATA_PROCESSED / "agentes_curvas.json").write_text(json.dumps(curves, indent=1), encoding="utf-8")
    print(f"  {len(profiles)} agent series")

    s = add_lags(series)
    print("TP2 regression")
    save(regression.run(s), RESULTS)
    print("TP3 classification and region-month clusters")
    save(classification.run(s), RESULTS)
    print("Done:", sorted(p.name for p in RESULTS.iterdir()))


if __name__ == "__main__":
    main()
