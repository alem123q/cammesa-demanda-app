"""Build the extended regional series (2017 to the latest annual report) in data/processed/.

    python scripts/build_extended.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cammesa.config import DATA_PROCESSED  # noqa: E402
from cammesa.data import extended_regional_series  # noqa: E402

s = extended_regional_series()
out = DATA_PROCESSED / "serie_regional_extendida.csv"
s.to_csv(out, index=False)
months = s["indice_tiempo"].nunique()
print(f"{len(s)} rows, {s['region'].nunique()} regions x {months} months "
      f"({s['indice_tiempo'].min()} to {s['indice_tiempo'].max()}) -> {out}")
print(s.groupby("fuente")["indice_tiempo"].agg(["min", "max", "nunique"]))
