"""Pre-registered evaluation on 2022-2025 (docs/preregistro_2022_2025.md).

    python scripts/evaluate_2022_2025.py --stage validation   # choose alpha per model, test untouched
    python scripts/evaluate_2022_2025.py --stage test         # fit on 2018-2021 and evaluate once
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from cammesa import extended as ex  # noqa: E402
from cammesa.config import RESULTS  # noqa: E402

pd.set_option("display.width", 200)
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["validation", "test"], required=True)
a = ap.parse_args()
RESULTS.mkdir(exist_ok=True)
D = ex.build_table()
missing = D.drop(columns=["demanda_MWh"]).isna().sum()
assert missing.sum() == 0, f"missing values:\n{missing[missing > 0]}"
train, test = ex.split(D)
print(f"train {len(train)} rows {train.indice_tiempo.min()}..{train.indice_tiempo.max()} | "
      f"test {len(test)} rows {test.indice_tiempo.min()}..{test.indice_tiempo.max()}")

if a.stage == "validation":
    v = ex.validate(D)
    v.to_csv(RESULTS / "ext_validacion.csv", index=False)
    alphas = ex.chosen_alphas(v)
    (RESULTS / "ext_alphas.json").write_text(json.dumps(alphas, indent=1, ensure_ascii=False), encoding="utf-8")
    show = pd.concat([v[v.alpha.isna()], v.dropna(subset=["alpha"]).loc[lambda x: x.groupby("modelo")["RMSE validación (GWh)"].transform("min") == x["RMSE validación (GWh)"]]])
    print(show.round(1).to_string(index=False))
    print("chosen alphas:", alphas)
else:
    alphas = json.loads((RESULTS / "ext_alphas.json").read_text(encoding="utf-8"))
    res, pred = ex.test(D, alphas)
    res.to_csv(RESULTS / "ext_prueba_metricas.csv", index=False)
    pred.to_csv(RESULTS / "ext_prueba_predicciones.csv", index=False)
    print(res.round(1).to_string(index=False))
