"""Pre-registered classification evaluation on 2022-2025 (docs/preregistro_clasificacion_2022_2025.md).

    python scripts/evaluate_classification_2022_2025.py --stage validation
    python scripts/evaluate_classification_2022_2025.py --stage test
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from cammesa import extended as ex  # noqa: E402
from cammesa.config import RESULTS  # noqa: E402

pd.set_option("display.width", 220)
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["validation", "test"], required=True)
a = ap.parse_args()
D = ex.build_table()
alphas = json.loads((RESULTS / "ext_alphas.json").read_text(encoding="utf-8"))

if a.stage == "validation":
    v, Cs = ex.validate_classification(D, alphas)
    v.to_csv(RESULTS / "ext_clf_validacion.csv", index=False)
    (RESULTS / "ext_clf_C.json").write_text(json.dumps(Cs, indent=1, ensure_ascii=False), encoding="utf-8")
    print(v.round(3).to_string(index=False))
    print("chosen C:", Cs)
else:
    Cs = json.loads((RESULTS / "ext_clf_C.json").read_text(encoding="utf-8"))
    res, pred = ex.test_classification(D, alphas, Cs)
    res.to_csv(RESULTS / "ext_clf_prueba_metricas.csv", index=False)
    pred.to_csv(RESULTS / "ext_clf_prueba_predicciones.csv", index=False)
    print(f"share of 'sube' in test: {pred['sube'].mean():.0%}")
    print(res.round(3).to_string(index=False))
