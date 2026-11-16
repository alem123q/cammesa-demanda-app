"""Más allá del TP · Clima, actividad y tarifas, y la prueba pre-registrada sobre 2022–2025."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cammesa import exploratory
from dashboard.common import GRIS, NARANJA, NOMBRES, PROC, TINTA, VERDE, caja, csv, jsn, layout, num, pct

AZUL = "#4F81BD"
REGLA = "Regla: mismo mes del año anterior"
NOMBRE = {REGLA: "Regla simple", "Ridge del TP2": "Ridge del TP2",
          "Modelo de pronóstico": "Modelo de pronóstico", "Modelo explicativo": "Modelo explicativo"}
COLOR = {REGLA: GRIS, "Ridge del TP2": "#A9C4E8", "Modelo de pronóstico": AZUL, "Modelo explicativo": NARANJA}

st.title("🌡️ Más allá del TP · 2017–2025")
st.caption("Continuación personal del proyecto: se suman clima, actividad económica y tarifas, se extienden los datos "
           "hasta 2025 y se ponen a prueba las conclusiones en cuatro años que nadie había mirado.")


@st.cache_data
def descomposicion(rezago):
    return exploratory.decomposition(rezago)


@st.cache_data
def serie_ext():
    return pd.read_csv(PROC / "serie_regional_extendida.csv", dtype={"indice_tiempo": str})


res = csv("ext_prueba_metricas.csv").set_index("modelo")
pred = csv("ext_prueba_predicciones.csv")
regla = res.loc[REGLA, "RMSE (GWh)"]

t1, t2, t3 = st.tabs(["Resultado de la prueba 2022–2025", "Mes a mes por región", "¿Por qué cayó la demanda en 2018–2019?"])

with t1:
    caja("**Cómo se hizo:** primero se escribió y subió a GitHub el plan (modelos, variables, métrica y metas); "
         "después se eligió el ajuste de cada modelo con datos de 2018–2021; recién al final se abrió la prueba "
         "2022–2025, una sola vez. El orden queda registrado en el historial del repositorio.")
    c = st.columns(3)
    for col, (k, txt, meta) in zip(c, [("Ridge del TP2", "Solo la historia de la demanda, como en el TP", None),
                                       ("Modelo de pronóstico", "Solo información disponible al pronosticar", 1),
                                       ("Modelo explicativo", "Con el clima y la actividad reales del mes", 2)]):
        r = res.loc[k, "RMSE (GWh)"]
        col.metric(NOMBRE[k] + " · RMSE (GWh)", num(r), pct((r / regla - 1) * 100) + " vs. regla", delta_color="inverse")
        if meta:
            ok = r <= 0.9 * regla
            col.caption(f"{txt}. Meta {meta} (−10 %): **{'cumplida ✅' if ok else 'no cumplida ❌'}**")
        else:
            col.caption(txt + ". Sin meta: es la referencia del TP.")
    st.caption(f"La regla simple («mismo mes del año anterior») tiene {num(regla)} GWh de RMSE en la prueba: "
               "432 casos, 9 regiones × 48 meses.")

    anios = ["2022", "2023", "2024", "2025"]
    fig = go.Figure([go.Bar(name=NOMBRE[k], x=anios, y=[res.loc[k, f"RMSE {a}"] for a in anios], marker_color=COLOR[k],
                            text=[f"{res.loc[k, f'RMSE {a}']:.0f}" for a in anios], textposition="outside")
                     for k in NOMBRE])
    layout(fig, 400, "RMSE por año (GWh, menos es mejor)")
    fig.update_yaxes(range=[0, max(res.loc[k, f"RMSE {a}"] for k in NOMBRE for a in anios) * 1.15])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- **El Ridge del TP vuelve a empatar con la regla**: con solo la historia de la demanda no se le gana.\n"
        "- **El clima, la actividad y las tarifas explican gran parte de lo que la regla no ve**: 37 % menos error.\n"
        "- **Para pronosticar de verdad**, sin conocer el clima del mes, la mejora es de 9 %: gana en 2022 y 2023, "
        "empata en 2024 y 2025, y queda apenas debajo de la meta.")
    st.subheader("Clasificación: ¿sube o baja frente al año anterior?")
    cl = csv("ext_clf_prueba_metricas.csv").set_index("modelo")
    pers = cl.loc["Regla: persistencia"]
    c = st.columns(3)
    for col, (k, txt) in zip(c, [("SVM del TP3", "Solo la historia de la demanda, como en el TP3. Sin meta."),
                                ("Clasificador de pronóstico", "«Sube» si el modelo de pronóstico prevé un aumento. Meta 1"),
                                ("Clasificador explicativo", "«Sube» si el modelo explicativo prevé un aumento. Meta 2")]):
        r = cl.loc[k]
        col.metric(k + " · F1 «sube»", num(r["F1 «sube»"], 2), num(r["F1 «sube»"] - pers["F1 «sube»"], 2) + " vs. persistencia")
        ok = r["F1 «sube»"] > pers["F1 «sube»"] and r["Kappa"] > pers["Kappa"]
        col.caption(f"Kappa {num(r['Kappa'], 2)} (persistencia {num(pers['Kappa'], 2)}). {txt}" +
                    ("" if k == "SVM del TP3" else f" (superar a la persistencia en F1 y kappa): **{'cumplida ✅' if ok else 'no cumplida ❌'}**"))
    st.caption(f"Persistencia: «sube» si el mes anterior subió (F1 {num(pers['F1 «sube»'], 2)}). «Siempre sube» tiene F1 "
               f"{num(cl.loc['Siempre «sube»', 'F1 «sube»'], 2)} porque el 56 % de los meses subió, pero kappa 0: nunca "
               "detecta una baja. Por eso la meta exige también kappa.")
    alphas = jsn("ext_alphas.json")
    with st.expander("Detalle de los modelos"):
        st.markdown(
            "| Modelo | Qué predice | Variables |\n|---|---|---|\n"
            "| Regla simple | — | la demanda del mismo mes del año anterior |\n"
            "| Ridge del TP2 | la demanda | región, mes anterior, mismo mes del año anterior, mes, trimestre |\n"
            "| Modelo de pronóstico | el cambio frente al año anterior | cambio del mes previo; clima normal del mes "
            "contra el real del año anterior (por región); EMAE publicado (hace 2 meses); tarifa real (hace 6 meses) |\n"
            "| Modelo explicativo | el cambio frente al año anterior | cambio del mes previo; clima real del mes contra "
            "el del año anterior (por región); EMAE del mes; tarifa real (hace 6 meses) |")
        st.caption("Ajuste elegido en validación (α de Ridge): " +
                   ", ".join(f"{NOMBRE[k]} {v:g}" for k, v in alphas.items()) + ".")

with t2:
    region = st.selectbox("Región", sorted(pred["region"].unique()), format_func=lambda r: NOMBRES.get(r, r),
                          index=sorted(pred["region"].unique()).index("GRAN BS.AS."))
    g = pred[pred["region"] == region]
    x = pd.PeriodIndex(g["indice_tiempo"], freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_scatter(x=x, y=g["demanda_MWh"] / 1000, name="Lo que pasó", line=dict(color=TINTA, width=3))
    fig.add_scatter(x=x, y=g[REGLA] / 1000, name="Regla simple", line=dict(color=GRIS, dash="dash"))
    fig.add_scatter(x=x, y=g["Modelo de pronóstico"] / 1000, name="Modelo de pronóstico", line=dict(color=AZUL))
    fig.add_scatter(x=x, y=g["Modelo explicativo"] / 1000, name="Modelo explicativo", line=dict(color=NARANJA))
    layout(fig, 420, f"{NOMBRES.get(region, region)} · demanda mensual (GWh)")
    st.plotly_chart(fig, use_container_width=True)
    err = {k: ((g[k] - g["demanda_MWh"]) ** 2).mean() ** 0.5 / 1000 for k in NOMBRE}
    st.caption("RMSE en esta región: " + " · ".join(f"{NOMBRE[k]} {num(v)} GWh" for k, v in err.items()))

    s = serie_ext()
    tot = s.groupby("indice_tiempo")["demanda_MWh"].sum() / 1e6
    fig = go.Figure(go.Scatter(x=pd.PeriodIndex(tot.index, freq="M").to_timestamp(), y=tot.values,
                               line=dict(color=TINTA, width=2), name="Demanda total"))
    fig.add_vrect(x0="2019-09-01", x1="2020-02-29", fillcolor=VERDE, opacity=0.12, line_width=0,
                  annotation_text="prueba del TP", annotation_position="top left")
    fig.add_vrect(x0="2022-01-01", x1="2025-12-31", fillcolor=AZUL, opacity=0.10, line_width=0,
                  annotation_text="prueba 2022–2025", annotation_position="top left")
    layout(fig, 320, "Demanda total del país, 2017–2025 (TWh)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("2017–2018: archivo de datos abiertos del TP. Desde 2019: bases de los informes anuales de CAMMESA "
               "(valores definitivos; coinciden con el archivo del TP salvo en sus últimos meses, que eran provisorios).")

with t3:
    caja("**Exploratorio:** usa el período del TP, que ya se había mirado. Sirvió para formular la prueba "
         "2022–2025, no es evidencia por sí solo.")
    d, coef = descomposicion(6)
    fila = d.iloc[0]
    partes = {"heating (cold)": "Frío (calefacción)", "cooling (heat)": "Calor (refrigeración)",
              "activity (EMAE)": "Actividad (EMAE)", "real tariff": "Tarifa real (hace 6 meses)", "rest": "Resto"}
    colores = [VERDE, VERDE, AZUL, NARANJA, GRIS]
    fig = go.Figure(go.Bar(x=list(partes.values()), y=[fila[k] for k in partes], marker_color=colores,
                           text=[pct(fila[k]).replace(" %", "") for k in partes], textposition="outside"))
    layout(fig, 380, f"Caída sep 2018 – ago 2019: {pct(fila['actual change'])} · aporte de cada causa (puntos)")
    fig.update_yaxes(range=[min(fila[k] for k in partes) - 0.8, 0.6])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- **La mitad de la caída fue clima**: veranos más suaves y un invierno de 2019 templado.\n"
        "- **Recesión y tarifas juntas, cerca de un tercio.** Se movieron a la vez, así que el reparto entre las dos "
        "es inestable; lo firme es su suma (unos −2 puntos). Las tarifas pesan con 6 a 12 meses de retraso.\n"
        "- **El resto** incluye casos puntuales, como Minera Alumbrera (cerca de medio punto).\n"
        "- **La recuperación de la prueba del TP** (sep 2019 – feb 2020) fue casi toda por el clima.")
