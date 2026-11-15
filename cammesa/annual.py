"""Read the demand detail from CAMMESA's annual report databases ("Base de datos Informe Anual YYYY").

Each annual file is a zip with one xlsx. Its DEMANDA sheet has summary tables at the top and,
further down, the detail table (one row per month, agent and tariff) for the report year and the
previous one. The detail has the same fields as the open-data file used in the course, so it is
returned with the same column names.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

from .config import DATA_RAW

ANNUAL_DIR = DATA_RAW / "cammesa"
FILLER_TARIFF = "MERCADO TERMINO DISTRIB"   # "Mercado a Término" rows in the open-data file

# header in the xlsx -> column name in the open-data file
COLUMNS = {"AÑO": "anio", "MES": "mes", "AGENTE NEMO": "agente_nemo", "AGENTE DESCRIPCION": "agente_descripcion",
           "TIPO AGENTE": "tipo_agente", "REGION": "region", "PROVINCIA": "provincia",
           "CATEGORIA AREA": "categoria_area", "CATEGORIA DEMANDA": "categoria_demanda",
           "CATEGORIA TARIFA": "categoria_tarifa", "TARIFA": "tarifa", "TIPO DEMANDA": "tipo_demanda",
           "DEMANDA [MWh]": "demanda_MWh"}
KEYS = {k.upper(): v for k, v in COLUMNS.items()}


def _workbook(path: Path) -> openpyxl.Workbook:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xlsm")))
            return openpyxl.load_workbook(io.BytesIO(z.read(name)), read_only=True, data_only=True)
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def read_annual(path: Path | str) -> pd.DataFrame:
    """Demand detail of one annual report file (zip or xlsx)."""
    wb = _workbook(Path(path))
    ws = wb["DEMANDA"]
    header, rows = None, []
    # The sheet declares 16384 columns; read only the first 20 and stream the rows.
    for row in ws.iter_rows(max_col=20, values_only=True):
        if header is None:
            cells = [str(c).strip().upper() if c is not None else "" for c in row]
            if "AGENTE NEMO" in cells and "AÑO" in cells:
                header = {i: KEYS[c] for i, c in enumerate(cells) if c in KEYS}
            continue
        if row[min(header)] is None and all(row[i] is None for i in header):
            continue
        rows.append([row[i] for i in header])
    wb.close()
    if header is None:
        raise ValueError(f"No detail table found in the DEMANDA sheet of {path}")
    d = pd.DataFrame(rows, columns=list(header.values())).dropna(subset=["agente_nemo"])
    month = pd.to_datetime(d["mes"])
    d["indice_tiempo"], d["anio"], d["mes"] = month.dt.strftime("%Y-%m"), month.dt.year, month.dt.month
    d["demanda_MWh"] = pd.to_numeric(d["demanda_MWh"], errors="coerce").fillna(0.0)
    # Same label as the open-data file, so the regional series drops these rows the same way
    d.loc[d["tarifa"] == FILLER_TARIFF, "categoria_tarifa"] = "Mercado a Término"
    d["fuente"] = Path(path).name
    d["anio_informe"] = report_year(path)
    return d


def report_year(path: Path | str) -> int:
    """Report year taken from the file name (the last 4-digit year in it)."""
    years = re.findall(r"(?:19|20)\d{2}", Path(path).stem)
    if not years:
        raise ValueError(f"Cannot tell the report year from the file name: {path}")
    return int(years[-1])


def read_all_annual(folder: Path = ANNUAL_DIR) -> pd.DataFrame:
    """Every annual file in ``folder``; when two files cover the same month, the newer report wins."""
    files = sorted(folder.glob("*.zip")) + sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No annual report files in {folder}")
    frames = [read_annual(f) for f in files]
    d = pd.concat(frames, ignore_index=True)
    newest = d.groupby("indice_tiempo")["anio_informe"].max()
    return d[d["anio_informe"] == d["indice_tiempo"].map(newest)].reset_index(drop=True)
