"""TP2 · Comparar modelos: tabla, validación contra prueba y pronósticos de cada modelo contra lo real."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import GRIS, NARANJA, NOMBRES, TINTA, VERDE, caja, corto, csv, layout, num

st.title("📊 TP2 · Comparar modelos")
st.caption("Seis modelos y tres reglas simples, evaluados con la misma validación y la misma prueba.")

t = csv("tp2_metricas.csv")
base = t.set_index("modelo").loc["Regla: mismo mes del año anterior", "RMSE (GWh)"]
meta_rmse = 0.9 * base

caja(f"**Qué mirar:** la vara es la regla «mismo mes del año anterior» ({num(base)} GWh). La meta era bajar a "
     f"{num(meta_rmse)} GWh. Ningún modelo llega; el bosque queda apenas debajo de la regla, dentro del ruido.")

# ------------------------------------------------------------- tabla
st.subheader("Tabla de resultados")
vista = t[["modelo", "RMSE validación (GWh)", "desvío validación", "RMSE entrenamiento (GWh)",
           "RMSE (GWh)", "MAE (GWh)", "MAPE (%)", "Sesgo (%)", "R²"]].rename(columns={
    "RMSE (GWh)": "RMSE prueba (GWh)", "MAE (GWh)": "MAE prueba (GWh)", "MAPE (%)": "MAPE prueba (%)",
    "Sesgo (%)": "Sesgo prueba (%)", "R²": "R² prueba"}).sort_values("RMSE prueba (GWh)")
st.dataframe(vista.style.format({c: "{:.1f}" for c in vista.columns if c not in ("modelo", "R² prueba")} | {"R² prueba": "{:.3f}"})
             .highlight_between(subset=["RMSE prueba (GWh)"], right=meta_rmse, color="#D8EFE9"),
             hide_index=True, use_container_width=True)
st.caption("RMSE: tamaño típico del error, en GWh (menos es mejor). MAE: error absoluto medio. MAPE: error en %. "
           "Sesgo negativo: el modelo se queda corto. El R² da 0,97–0,996 para todos porque lo explica el tamaño de "
           "las regiones: no sirve para comparar.")

# ------------------------------------------------------------- validación vs prueba
st.subheader("¿La validación anticipó la prueba?")
orden = t.sort_values("RMSE validación (GWh)")
fig = go.Figure([
    go.Bar(name="Validación (promedio de 4 bloques)", x=orden["modelo"].map(corto), y=orden["RMSE validación (GWh)"], marker_color=VERDE,
           text=orden["RMSE validación (GWh)"].round(0), textposition="outside"),
    go.Bar(name="Prueba", x=orden["modelo"].map(corto), y=orden["RMSE (GWh)"], marker_color=NARANJA,
           text=orden["RMSE (GWh)"].round(0), textposition="outside")])
fig.add_hline(y=meta_rmse, line_dash="dash", line_color=TINTA, annotation_text="meta", annotation_position="top left")
fig.update_layout(barmode="group", yaxis_title="RMSE (GWh)", yaxis_range=[0, orden[["RMSE validación (GWh)", "RMSE (GWh)"]].max().max() * 1.12])
st.plotly_chart(layout(fig, 430), use_container_width=True)
from scipy.stats import spearmanr  # noqa: E402
rho = spearmanr(t["RMSE validación (GWh)"], t["RMSE (GWh)"]).statistic
mods = t[t["tipo"] == "modelo"]
rho_m = spearmanr(mods["RMSE validación (GWh)"], mods["RMSE (GWh)"]).statistic
caja(f"Ordenados de mejor a peor según la validación. Si la validación anticipara la prueba, las barras naranjas "
     f"subirían de izquierda a derecha igual que las verdes; pasa casi lo contrario. La correlación de rangos entre "
     f"los dos órdenes es {num(rho, 2)} con las nueve opciones, y {num(rho_m, 2)} entre los seis modelos: su orden se "
     f"invierte por completo. La validación cayó en el año de caída y la prueba en la recuperación.", "aviso")

with st.expander("Ver el error de cada bloque de validación"):
    b = t.set_index("modelo")[[f"RMSE bloque {k}" for k in range(1, 5)]]
    b.columns = ["sep–nov 2018", "dic 2018–feb 2019", "mar–may 2019", "jun–ago 2019"]
    st.dataframe(b.style.format("{:.1f}"), use_container_width=True)

# ------------------------------------------------------------- pronósticos contra lo real
st.subheader("Pronósticos de un modelo contra lo real")
c1, c2 = st.columns(2)
modelos = list(t["modelo"])
m = c1.selectbox("Modelo", modelos, index=modelos.index("Ridge (pre-registrado)"))
reg = c2.selectbox("Región", list(NOMBRES), format_func=NOMBRES.get, index=4)

pe, pv, pp = csv("tp2_pred_entrenamiento.csv"), csv("tp2_pred_validacion.csv"), csv("tp2_pred_prueba.csv")
f = lambda d: d[d["region"] == reg].assign(fecha=lambda x: pd.PeriodIndex(x["indice_tiempo"], freq="M").to_timestamp())
e_, v_, p_ = f(pe), f(pv), f(pp)
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.concat([e_, p_])["fecha"], y=pd.concat([e_, p_])["demanda_MWh"] / 1000, name="Real",
                         mode="lines+markers", line=dict(color=TINTA, width=2), marker=dict(size=4)))
fig.add_trace(go.Scatter(x=v_["fecha"], y=v_[m] / 1000, name="Pronóstico en validación", mode="lines+markers",
                         line=dict(color=VERDE, dash="dot"), marker=dict(size=5)))
fig.add_trace(go.Scatter(x=p_["fecha"], y=p_[m] / 1000, name="Pronóstico en la prueba", mode="lines+markers",
                         line=dict(color=NARANJA, width=3), marker=dict(size=7)))
fig.add_vrect(x0=pd.Timestamp("2019-09-01"), x1=pd.Timestamp("2020-02-28"), fillcolor=NARANJA, opacity=0.08, line_width=0,
              annotation_text="prueba", annotation_position="top left")
fig.add_vrect(x0=pd.Timestamp("2018-09-01"), x1=pd.Timestamp("2019-08-31"), fillcolor=VERDE, opacity=0.05, line_width=0,
              annotation_text="validación", annotation_position="top left")
fig.update_yaxes(title="GWh"); fig.update_xaxes(tickformat="%m/%Y")
st.plotly_chart(layout(fig, 420, f"{m} · {NOMBRES[reg]}"), use_container_width=True)
err = (p_[m] - p_["demanda_MWh"]) / p_["demanda_MWh"] * 100
st.markdown(f"En la prueba, en {NOMBRES[reg]}: error medio **{num(np.abs(err).mean())} %**, "
            f"{'se queda corto' if err.mean() < 0 else 'se pasa'} en promedio un **{num(abs(err.mean()))} %**.")
