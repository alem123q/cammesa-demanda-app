"""TP3 · Comparar modelos: tabla, F1 en validación y prueba, sobreajuste y matrices de confusión."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from comun import GRIS, NARANJA, TINTA, VERDE, caja, corto, csv, layout, num

st.title("🧮 TP3 · Comparar modelos")
st.caption("Ocho clasificadores (con búsqueda de hiperparámetros), la SVM elegida de antemano y dos reglas simples.")

t = csv("tp3_metricas.csv")
PERS = "Regla: persistencia (repetir el mes anterior)"
caja("**Qué mirar:** la regla «repetir el mes anterior» (persistencia) le gana a todos. En la prueba, todos los modelos "
     "tienen AUC menor que 0,5 y kappa negativa: ordenan peor que el azar.")

# ------------------------------------------------------------- tabla
st.subheader("Tabla de resultados")
conj = st.radio("Mostrar métricas de", ["prueba", "validación", "entrenamiento"], horizontal=True)
cols = ["F1 «sube»", "Precisión «sube»", "Exhaustividad «sube»", "Exactitud", "F1 macro", "Kappa", "AUC ROC", "Precisión media (PR)"]
vista = t[["modelo"] + [f"{c} · {conj}" for c in cols if f"{c} · {conj}" in t.columns]].copy()
vista.columns = [c.replace(f" · {conj}", "") for c in vista.columns]
vista = vista.sort_values("F1 «sube»", ascending=False)
st.dataframe(vista.style.format({c: "{:.2f}" for c in vista.columns if c != "modelo"}, na_rep="—"),
             hide_index=True, use_container_width=True)
st.caption("La exactitud engaña: en la prueba, decir siempre «sube» acierta el 85 %. Por eso la métrica principal es "
           "el F1 de «sube». «SVM pre-registrada» es la que elegimos antes de abrir la prueba; «SVM (con búsqueda)», la "
           "ajustada después, informada como análisis.")

# ------------------------------------------------------------- F1 validación y prueba
st.subheader("F1 de «sube»: validación contra prueba")
o = t.sort_values("F1 «sube» · prueba", ascending=False)
fig = go.Figure([go.Bar(name="Validación", x=o["modelo"].map(corto), y=o["F1 «sube» · validación"], marker_color=VERDE,
                        text=o["F1 «sube» · validación"].round(2), textposition="outside"),
                 go.Bar(name="Prueba", x=o["modelo"].map(corto), y=o["F1 «sube» · prueba"], marker_color=NARANJA,
                        text=o["F1 «sube» · prueba"].round(2), textposition="outside")])
fig.update_layout(barmode="group", yaxis_title="F1 «sube»", yaxis_range=[0, 1.05])
st.plotly_chart(layout(fig, 430), use_container_width=True)

# ------------------------------------------------------------- sobreajuste
st.subheader("¿Sobreajustan?")
mods = t[t["tipo"] == "modelo"].copy()
mods["brecha"] = mods["F1 «sube» · entrenamiento"] - mods["F1 «sube» · validación"]
mods = mods.sort_values("brecha")
fs = go.Figure([go.Bar(name="Entrenamiento", y=mods["modelo"], x=mods["F1 «sube» · entrenamiento"], orientation="h", marker_color=GRIS),
                go.Bar(name="Validación", y=mods["modelo"], x=mods["F1 «sube» · validación"], orientation="h", marker_color=VERDE)])
fs.update_layout(barmode="group", xaxis_title="F1 «sube»", xaxis_range=[0, 1])
st.plotly_chart(layout(fs, 420), use_container_width=True)
caja(f"Todos aprenden mucho mejor los datos que ya vieron (F1 {num(mods['F1 «sube» · entrenamiento'].min(), 2)}–"
     f"{num(mods['F1 «sube» · entrenamiento'].max(), 2)}) que los nuevos ({num(mods['F1 «sube» · validación'].min(), 2)}–"
     f"{num(mods['F1 «sube» · validación'].max(), 2)}). Esa brecha es sobreajuste. Los ensambles (bosque, gradient boosting, "
     "votación) no la cierran: reducen la varianza, pero el problema de fondo es de sesgo.")

# ------------------------------------------------------------- matrices
st.subheader("Matrices de confusión en la prueba (umbral 0,5)")
pp = csv("tp3_prob_prueba.csv")
todos = list(t["modelo"])
elegidos = st.multiselect("Modelos", todos, default=[PERS, "Regresión logística", "Árbol de decisión", "SVM pre-registrada (por defecto)"])
if elegidos:
    n = len(elegidos); cols_m = min(n, 4); filas = int(np.ceil(n / cols_m))
    fig = make_subplots(rows=filas, cols=cols_m, subplot_titles=[corto(m) for m in elegidos], horizontal_spacing=0.08, vertical_spacing=0.18)
    for i, m in enumerate(elegidos):
        cm = confusion_matrix(pp["sube"], pp[m + " · clase"], labels=[0, 1])
        fig.add_trace(go.Heatmap(z=[[1, 0], [0, 1]], x=["dijo baja", "dijo sube"], y=["era baja", "era sube"], text=cm,
                                 texttemplate="%{text}", textfont=dict(size=18), showscale=False,
                                 colorscale=[[0, "#FBEBD2"], [1, "#D8EFE9"]]), row=i // cols_m + 1, col=i % cols_m + 1)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(layout(fig, 290 * filas, leyenda_abajo=False), use_container_width=True)
    st.caption("Verde: aciertos (diagonal). Naranja: errores. En la prueba hubo 46 «sube» y 8 «baja». El error dominante "
               "es «subió y dijo baja»: los modelos aprendieron que casi todo bajaba.")
