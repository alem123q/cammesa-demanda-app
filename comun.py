"""Funciones y datos compartidos por todas las páginas del tablero."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"

VERDE, NARANJA, GRIS, TINTA, GRIS_CLARO = "#2A7F86", "#E89B25", "#9AA3AE", "#1C2433", "#EEF1F4"
PALETA = [VERDE, NARANJA, "#6B5CA5", "#C0504D", "#4F81BD", "#8064A2", "#9BBB59", "#F79646"]

NOMBRES = {"BUENOS AIRES": "Buenos Aires (interior)", "CENTRO": "Centro", "COMAHUE": "Comahue", "CUYO": "Cuyo",
           "GRAN BS.AS.": "Gran Buenos Aires", "LITORAL": "Litoral", "NORESTE": "Noreste",
           "NOROESTE": "Noroeste", "PATAGONICA": "Patagónica"}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
MESES_CORTOS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# ------------------------------------------------------------------ carga (una vez por servidor)
@st.cache_resource
def modelo_ridge():
    return joblib.load(RAIZ / "modelo" / "ridge_regional.joblib")


@st.cache_data
def meta() -> dict:
    return json.loads((RAIZ / "modelo" / "metadatos.json").read_text(encoding="utf-8"))


@st.cache_data
def serie() -> pd.DataFrame:
    s = pd.read_csv(RAIZ / "datos" / "serie_regional.csv", dtype={"indice_tiempo": str})
    s["periodo"] = pd.PeriodIndex(s["indice_tiempo"], freq="M")
    return s


@st.cache_data
def csv(nombre: str) -> pd.DataFrame:
    ruta = RES / nombre if (RES / nombre).exists() else RAIZ / "datos" / nombre
    return pd.read_csv(ruta, dtype={"indice_tiempo": str})


@st.cache_data
def jsn(nombre: str):
    ruta = RES / nombre if (RES / nombre).exists() else RAIZ / "datos" / nombre
    return json.loads(ruta.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ formato (coma decimal)
def num(x: float, dec: int = 1) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def gwh(mwh: float) -> str:
    return num(mwh / 1000) + " GWh"


def pct(x: float, signo: bool = True, dec: int = 1) -> str:
    return (f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}").replace(".", ",") + " %"


def periodo_txt(p: str) -> str:
    a, m = p.split("-")[:2]
    return f"{MESES[int(m) - 1]} {a}"


def layout(fig, alto=380, titulo=None, leyenda_abajo=True):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50 if titulo else 20, b=10), plot_bgcolor="white",
                      title_text=titulo or "", font=dict(size=13),
                      legend=dict(orientation="h", y=-0.2) if leyenda_abajo else None)
    fig.update_yaxes(gridcolor="#E3E7EB", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


CORTOS = {"Regla: mismo mes del año anterior": "Regla: año anterior", "Regla: media de la región": "Regla: media región",
          "Regla: persistencia (repetir el mes anterior)": "Persistencia", "SVM pre-registrada (por defecto)": "SVM pre-registrada",
          "Regla: siempre «baja»": "Siempre «baja»", "Ridge (pre-registrado)": "Ridge (pre-reg.)"}


def corto(nombre: str) -> str:
    """Nombre corto para ejes y títulos de gráficos."""
    return CORTOS.get(nombre, nombre)


def caja(texto: str, tipo: str = "info"):
    """Recuadro de lectura rápida, para que cada página diga en una frase qué hay que mirar."""
    {"info": st.info, "ok": st.success, "aviso": st.warning}[tipo](texto)
