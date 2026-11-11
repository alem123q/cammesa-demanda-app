"""Inicio: qué resuelve el proyecto, los resultados en tres números y cómo leer el tablero."""

import pandas as pd
import streamlit as st

from comun import csv, num

st.title("⚡ ¿Se puede predecir la demanda eléctrica de cada región?")
st.markdown(
    "Usamos los datos públicos de **CAMMESA** (la empresa que administra el mercado eléctrico mayorista) "
    "con la demanda de energía de las **9 regiones** del país, mes a mes, de **enero 2017 a febrero 2020**. "
    "Con eso respondimos dos preguntas:")

c1, c2 = st.columns(2)
with c1:
    st.subheader("TP2 · ¿Cuántos GWh?")
    st.markdown("Predecir **un número**: cuánta energía va a consumir cada región el mes que viene.")
with c2:
    st.subheader("TP3 · ¿Sube o baja?")
    st.markdown("Predecir **una categoría**: si la región va a consumir **más o menos** que el mismo mes del año anterior.")

t2, t3 = csv("tp2_metricas.csv").set_index("modelo"), csv("tp3_metricas.csv").set_index("modelo")
st.divider()
st.subheader("El resultado en tres números")
k1, k2, k3 = st.columns(3)
k1.metric("Error de copiar el año anterior (regla simple)", num(t2.loc["Regla: mismo mes del año anterior", "RMSE (GWh)"]) + " GWh",
          help="Cuánto se equivoca, típicamente, la regla más simple en cada región y mes de la prueba.")
k2.metric("Error del modelo elegido (Ridge)", num(t2.loc["Ridge (pre-registrado)", "RMSE (GWh)"]) + " GWh",
          delta="no llega a la meta de 63,9", delta_color="off")
k3.metric("Aciertos de «repetir el mes anterior» (F1)",
          num(t3.loc["Regla: persistencia (repetir el mes anterior)", "F1 «sube» · prueba"], 2),
          delta="le gana a los 8 clasificadores", delta_color="off")
st.success("**En una frase:** con solo el historial de la demanda, copiar el pasado no se supera de forma confiable. "
           "Los modelos fallan cuando cambia la tendencia: aprendieron con la demanda cayendo (sep 2018 – ago 2019) "
           "y se los evaluó cuando volvía a crecer.")

st.divider()
st.subheader("Cómo leer los números")
st.markdown("""
| Término | Qué significa | Ejemplo |
|---|---|---|
| **GWh** | Gigavatio-hora, una unidad de energía. 1 GWh = 1.000 MWh. | Una región consume entre 370 y 5.250 GWh por mes. |
| **RMSE** | El tamaño típico del error, en GWh. **Menos es mejor.** Castiga más los errores grandes. | 71 GWh: la regla se equivoca unos 71 GWh por región y mes. |
| **Entrenar / probar** | Entrenar es aprender con el pasado; probar es medir con meses que el modelo nunca vio. | Entrena hasta ago 2019; prueba de sep 2019 a feb 2020. |
| **Validación** | «Mini-exámenes» dentro del entrenamiento para elegir el modelo sin mirar la prueba. | 4 bloques de 3 meses, de sep 2018 a ago 2019. |
| **F1 de «sube»** | Qué tan bien detecta los meses que suben, sin inventarlos. Va de 0 a 1. | 0,83 la persistencia; 0,70 el mejor modelo. |
| **Umbral** | Desde qué probabilidad el modelo dice «sube». Debajo, lo descarta y dice «baja». | 0,5 por defecto; se puede mover en «Umbral y curvas». |
""")

st.subheader("Las variables que usa cada modelo")
st.dataframe(pd.DataFrame([
    ["Región (9)", "✓", "✓", "Una columna 0/1 por región"],
    ["Demanda del mes anterior", "✓", "✓", "Logaritmo y normalizada (media 0, desvío 1)"],
    ["Demanda del mismo mes del año anterior", "✓", "✓", "Logaritmo y normalizada"],
    ["Mes y trimestre", "✓", "✓", "Normalizados"],
    ["Variación interanual del mes anterior", "—", "✓", "Normalizada"],
    ["Tarifa y categoría", "✗", "✗", "Excluidas: en feb 2019 cambió el registro, no el consumo"],
], columns=["Variable", "TP2", "TP3", "Tratamiento"]), hide_index=True, use_container_width=True)
st.caption("La normalización se ajusta solo con el entrenamiento, dentro de un Pipeline: la validación y la prueba "
           "no le pasan información al modelo. Los árboles no la necesitan y usan los valores sin escalar.")

st.subheader("Por dónde empezar")
st.markdown("""
- **¿Querés un pronóstico?** → *TP2 · Probar el pronóstico*: elegí región y mes.
- **¿Qué modelo anduvo mejor?** → *TP2 · Comparar modelos* y *TP3 · Comparar modelos*.
- **¿Qué pasa si saco una variable?** → *TP2 · Jugar con las variables*.
- **¿Cuándo el modelo dice «sube» y cuándo lo descarta?** → *TP3 · Umbral y curvas*.
- **¿En qué grupo está mi región o mi empresa?** → *Agrupamiento · Grupos y buscador*.
""")
