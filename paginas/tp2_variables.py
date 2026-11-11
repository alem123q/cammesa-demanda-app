"""TP2 · Jugar con las variables: reentrena Ridge en el momento con las variables y opciones elegidas."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from comun import GRIS, NARANJA, TINTA, VERDE, caja, csv, layout, num
from preparar_resultados import CV, particion, serie_con_rezagos

st.title("🎛️ TP2 · Jugar con las variables")
st.caption("Sacá o agregá variables, prendé o apagá la normalización y mové la regularización: "
           "el modelo se vuelve a entrenar al instante y te muestra si mejora o empeora.")

VARS = {"region": "Región", "rezago_1": "Demanda del mes anterior", "rezago_12": "Demanda del mismo mes del año anterior",
        "mes": "Mes", "trimestre": "Trimestre"}


@st.cache_data
def datos():
    e, t = particion(serie_con_rezagos(), ["rezago_1", "rezago_12"])
    return e, t


@st.cache_data(show_spinner="Entrenando…")
def evaluar(vars_: tuple, normalizar: bool, tipo: str, alpha: float):
    e, t = datos()
    partes = []
    if "region" in vars_:
        partes.append(("region", OneHotEncoder(drop="first", handle_unknown="ignore"), ["region"]))
    rez = [v for v in ("rezago_1", "rezago_12") if v in vars_]
    if rez:
        partes.append(("rezagos", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                            ("esc", StandardScaler() if normalizar else "passthrough")]), rez))
    cal = [v for v in ("mes", "trimestre") if v in vars_]
    if cal:
        partes.append(("calendario", StandardScaler() if normalizar else "passthrough", cal))
    mod = Ridge(alpha=alpha) if tipo == "Ridge" else LinearRegression()
    est = TransformedTargetRegressor(regressor=Pipeline([("prep", ColumnTransformer(partes)), ("modelo", mod)]),
                                     func=np.log1p, inverse_func=np.expm1)
    cols = list(vars_)
    rmse = lambda y, p: float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) / 1000)
    val = [rmse(e.iloc[b]["demanda_MWh"], clone(est).fit(e.iloc[a][cols], e.iloc[a]["demanda_MWh"]).predict(e.iloc[b][cols]))
           for a, b in CV.split(e)]
    final = clone(est).fit(e[cols], e["demanda_MWh"])
    nombres = final.regressor_.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(final.regressor_.named_steps["modelo"].coef_, index=nombres)
    return {"val": float(np.mean(val)), "val_blq": val, "ent": rmse(e["demanda_MWh"], final.predict(e[cols])),
            "pru": rmse(t["demanda_MWh"], final.predict(t[cols])), "coef": coef}


ref = csv("tp2_metricas.csv").set_index("modelo")
base_pru, base_val = ref.loc["Regla: mismo mes del año anterior", "RMSE (GWh)"], ref.loc["Regla: mismo mes del año anterior", "RMSE validación (GWh)"]
ridge_pru, ridge_val = ref.loc["Ridge (pre-registrado)", "RMSE (GWh)"], ref.loc["Ridge (pre-registrado)", "RMSE validación (GWh)"]

# ------------------------------------------------------------- controles
with st.container(border=True):
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Variables**")
        elegidas = tuple(v for v, txt in VARS.items() if st.checkbox(txt, value=True, key=f"v_{v}"))
    with c2:
        st.markdown("**Opciones**")
        tipo = st.radio("Modelo", ["Ridge", "Regresión lineal (sin regularizar)"], horizontal=True, key="tipo")
        normalizar = st.toggle("Normalizar (media 0, desvío 1)", value=True, key="normalizar",
                               help="El escalado se ajusta solo con los datos de entrenamiento.")
        alpha = st.select_slider("Regularización α de Ridge", options=[0.01, 0.1, 0.32, 1.0, 3.16, 10.0, 31.6, 100.0, 1000.0],
                                 value=3.16, disabled=tipo != "Ridge", key="alpha",
                                 help="Cuánto se frena al modelo. Más α: coeficientes más chicos y un modelo más simple. "
                                      "El elegido en el TP2 es 3,16.")
    def reiniciar():
        for v in VARS:
            st.session_state[f"v_{v}"] = True
        st.session_state["normalizar"], st.session_state["alpha"], st.session_state["tipo"] = True, 3.16, "Ridge"
    st.button("Volver a la configuración del TP2", on_click=reiniciar)

if not elegidas:
    st.error("Elegí al menos una variable.")
    st.stop()

r = evaluar(elegidas, normalizar, "Ridge" if tipo == "Ridge" else "Lineal", alpha)

# ------------------------------------------------------------- resultado
k1, k2, k3 = st.columns(3)
k1.metric("Error en validación (elige el modelo)", f"{num(r['val'])} GWh", delta=f"{num(r['val'] - ridge_val)} vs. Ridge del TP2",
          delta_color="inverse")
k2.metric("Error en la prueba", f"{num(r['pru'])} GWh", delta=f"{num(r['pru'] - base_pru)} vs. regla simple", delta_color="inverse")
k3.metric("Error en entrenamiento", f"{num(r['ent'])} GWh",
          help="Si es mucho menor que el de validación, el modelo memoriza (sobreajuste).")

if r["pru"] < 0.9 * base_pru:
    caja(f"Con esta configuración llegarías a la meta en la prueba ({num(0.9 * base_pru)} GWh). Ojo: elegir mirando la "
         "prueba es hacer trampa; la decisión se toma con la validación.", "aviso")
elif r["val"] < ridge_val - 1:
    caja("Mejora en validación frente al Ridge del TP2. Mirá si eso se sostiene en la prueba: es lo que nos pasó con "
         "otros modelos.", "info")

fig = go.Figure([
    go.Bar(name="Tu configuración", x=["Validación", "Prueba"], y=[r["val"], r["pru"]], marker_color=VERDE,
           text=[num(r["val"]), num(r["pru"])], textposition="outside"),
    go.Bar(name="Ridge del TP2", x=["Validación", "Prueba"], y=[ridge_val, ridge_pru], marker_color=GRIS,
           text=[num(ridge_val), num(ridge_pru)], textposition="outside"),
    go.Bar(name="Regla simple (año anterior)", x=["Validación", "Prueba"], y=[base_val, base_pru], marker_color=NARANJA,
           text=[num(base_val), num(base_pru)], textposition="outside")])
fig.update_layout(barmode="group", yaxis_title="RMSE (GWh)", yaxis_range=[0, max(r["val"], r["pru"], ridge_val, base_val) * 1.15])
st.plotly_chart(layout(fig, 380), use_container_width=True)

# ------------------------------------------------------------- coeficientes
with st.expander("¿Cuánto pesa cada variable? (coeficientes)"):
    c = r["coef"].rename(lambda n: n.split("__")[-1].replace("region_", "región: ")
                         .replace("rezago_12", "año anterior").replace("rezago_1", "mes anterior"))
    fc = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=[VERDE if v >= 0 else NARANJA for v in c.values]))
    st.plotly_chart(layout(fc, 60 + 26 * len(c), leyenda_abajo=False), use_container_width=True)
    st.caption("Los coeficientes están en escala logarítmica. Con las variables normalizadas se pueden comparar entre sí: "
               "cuanto más larga la barra, más pesa. Sin normalizar, dependen de las unidades de cada variable.")

with st.expander("¿Qué conviene probar?"):
    st.markdown("""
- **Sacar el mes y el trimestre.** Mejora un poco en validación (98,8 contra 103,7 GWh) y empeora en la prueba (80,2 contra 76,7): la estacionalidad ya entra por la demanda del mismo mes del año anterior, así que aportan poco y su efecto depende del período.
- **Sacar la región.** Casi no cambia en validación (102,2) y empeora en la prueba (84,0): las demandas pasadas ya llevan el nivel de cada región, pero la región ayuda a ajustarlo.
- **Quedarse solo con la región y la demanda del año anterior.** Es casi la regla simple escrita como modelo. Sale la mejor en validación (96,5) y una de las peores en la prueba (112,8): otro caso de validación engañosa.
- **Apagar la normalización con Ridge.** El resultado cambia, porque la penalización depende de la escala de cada variable. Con la regresión lineal sin regularizar, normalizar no cambia los pronósticos.
- **Mover α.** Con α muy grande el modelo se aplana y empeora (subajuste). Con α chico se parece a la regresión lineal.
""")
st.caption("Esto es exploración: el modelo oficial del TP2 sigue siendo Ridge con α = 3,16 y todas las variables, "
           "elegido antes de abrir la prueba.")
