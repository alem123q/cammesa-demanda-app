"""TP3 · Probar «sube o baja»: para una región y un mes, qué dijo cada modelo y qué pasó en realidad."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import GRIS, NARANJA, NOMBRES, TINTA, VERDE, caja, csv, layout, num, pct, periodo_txt

st.title("🔮 TP3 · Probar «sube o baja»")
st.caption("¿El mes que viene la región va a consumir más (sube) o menos (baja) que el mismo mes del año anterior?")

pv, pp = csv("tp3_prob_validacion.csv").assign(conjunto="validación"), csv("tp3_prob_prueba.csv").assign(conjunto="prueba")
todo = pd.concat([pv, pp], ignore_index=True)
MODELOS = [c for c in pp.columns if c.endswith(" · clase") and not c.startswith("Regla")]
MODELOS = [c.replace(" · clase", "") for c in MODELOS]
PERS = "Regla: persistencia (repetir el mes anterior)"

c1, c2 = st.columns(2)
reg = c1.selectbox("Región", list(NOMBRES), format_func=NOMBRES.get, index=4)
periodos = sorted(todo["indice_tiempo"].unique())
per = c2.selectbox("Mes a pronosticar", periodos, index=periodos.index("2019-10"), format_func=periodo_txt)
f = todo[(todo["region"] == reg) & (todo["indice_tiempo"] == per)].iloc[0]

real_sube = int(f["sube"]) == 1
var_real = (f["demanda_MWh"] / f["rezago_12"] - 1) * 100
pers_sube = int(f[PERS + " · clase"]) == 1

st.caption(f"Este mes es parte de la **{f['conjunto']}**: ningún modelo lo vio al entrenar. "
           + ("Los meses de validación (sep 2018 – ago 2019) son del año de caída." if f["conjunto"] == "validación"
              else "Los meses de prueba (sep 2019 – feb 2020) son de la recuperación."))

k1, k2, k3 = st.columns(3)
k1.metric("Lo que pasó", "SUBE ⬆️" if real_sube else "BAJA ⬇️", delta=f"{pct(var_real)} frente al año anterior", delta_color="off")
k2.metric("Regla «repetir el mes anterior»", "SUBE ⬆️" if pers_sube else "BAJA ⬇️",
          delta="acierta ✅" if pers_sube == real_sube else "falla ❌", delta_color="off",
          help=f"El mes anterior había variado {pct(f['var_previa'] * 100)} frente al año anterior.")
aciertos = sum(int(f[m + " · clase"]) == int(f["sube"]) for m in MODELOS)
k3.metric("Modelos que aciertan", f"{aciertos} de {len(MODELOS)}")

# ------------------------------------------------------------- probabilidades
probs = pd.Series({m: f[m] for m in MODELOS}).sort_values()
colores = [VERDE if int(f[m + " · clase"]) == int(f["sube"]) else NARANJA for m in probs.index]
fig = go.Figure(go.Bar(x=probs.values, y=probs.index, orientation="h", marker_color=colores,
                       text=[num(v, 2) for v in probs.values], textposition="outside"))
fig.add_vline(x=0.5, line_dash="dash", line_color=TINTA, annotation_text="umbral 0,5", annotation_position="top")
fig.update_xaxes(range=[0, 1.08], title="Probabilidad de «sube» que da el modelo")
st.plotly_chart(layout(fig, 70 + 34 * len(probs), "¿Qué dijo cada modelo?", leyenda_abajo=False), use_container_width=True)
caja("Cada barra es la probabilidad de «sube». A la derecha del umbral, el modelo dice «sube»; a la izquierda, lo "
     "descarta y dice «baja». **Verde:** acertó. **Naranja:** se equivocó. (La SVM decide por su margen interno, "
     "por eso puede no coincidir exactamente con el umbral.)")

# ------------------------------------------------------------- la serie de la región
with st.expander("Ver toda la serie de la región: ¿cuándo subió y cuándo bajó?"):
    r = todo[todo["region"] == reg].sort_values("indice_tiempo")
    var = (r["demanda_MWh"] / r["rezago_12"] - 1) * 100
    fx = go.Figure(go.Bar(x=pd.PeriodIndex(r["indice_tiempo"], freq="M").to_timestamp(), y=var,
                          marker_color=[VERDE if v > 0 else NARANJA for v in var]))
    fx.add_vrect(x0=pd.Timestamp("2019-08-16"), x1=pd.Timestamp("2020-02-15"), fillcolor=GRIS, opacity=0.12, line_width=0,
                 annotation_text="prueba", annotation_position="top left")
    fx.update_yaxes(title="Variación frente al año anterior (%)"); fx.update_xaxes(tickformat="%m/%Y")
    st.plotly_chart(layout(fx, 330, f"{NOMBRES[reg]}: verde = sube, naranja = baja", leyenda_abajo=False), use_container_width=True)
    st.caption("La persistencia acierta casi siempre porque los colores vienen en rachas: el signo cambia pocas veces al año.")
