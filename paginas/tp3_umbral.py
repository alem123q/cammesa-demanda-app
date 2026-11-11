"""TP3 · Umbral y curvas: mover el umbral y ver cómo cambian la matriz de confusión y las métricas."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

from comun import GRIS, NARANJA, TINTA, VERDE, caja, csv, layout, num

st.title("🎚️ TP3 · Umbral y curvas")
st.caption("¿Desde qué probabilidad el modelo dice «sube»? Debajo del umbral lo descarta y dice «baja».")

pv, pp = csv("tp3_prob_validacion.csv"), csv("tp3_prob_prueba.csv")
MODELOS = [c.replace(" · clase", "") for c in pp.columns if c.endswith(" · clase") and not c.startswith("Regla")]

c1, c2 = st.columns([1.4, 1])
m = c1.selectbox("Modelo", MODELOS, index=MODELOS.index("Regresión logística"))
conj = c2.radio("Datos", ["Prueba (sep 2019 – feb 2020)", "Validación (sep 2018 – ago 2019)"], horizontal=False)
d = pp if conj.startswith("Prueba") else pv
y, p = d["sube"].to_numpy(), d[m].to_numpy()
u = st.slider("Umbral: desde qué probabilidad dice «sube»", 0.0, 1.0, 0.5, 0.01)
pred = (p >= u).astype(int)

# ------------------------------------------------------------- matriz y métricas
(tn, fp), (fn, tp) = confusion_matrix(y, pred, labels=[0, 1])
c1, c2 = st.columns([1, 1.25])
with c1:
    z = [[tn, fp], [fn, tp]]
    txt = [[f"{tn}<br>bajó y dijo «baja» ✅", f"{fp}<br>bajó y dijo «sube» ❌"],
           [f"{fn}<br>subió y dijo «baja» ❌", f"{tp}<br>subió y dijo «sube» ✅"]]
    fig = go.Figure(go.Heatmap(z=[[1, 0], [0, 1]], x=["dijo «baja»", "dijo «sube»"], y=["era «baja»", "era «sube»"],
                               text=txt, texttemplate="%{text}", textfont=dict(size=14),
                               colorscale=[[0, "#FBEBD2"], [1, "#D8EFE9"]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(layout(fig, 330, f"Matriz de confusión · umbral {num(u, 2)}", leyenda_abajo=False), use_container_width=True)
with c2:
    prec, rec, f1 = precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0), f1_score(y, pred, zero_division=0)
    a, b = st.columns(2)
    a.metric("Precisión «sube»", num(prec, 2), help="De los que el modelo llamó «sube», cuántos subieron de verdad.")
    b.metric("Exhaustividad «sube»", num(rec, 2), help="De los que subieron, cuántos detectó el modelo.")
    a.metric("F1 «sube»", num(f1, 2), help="Combina precisión y exhaustividad. Va de 0 a 1.")
    b.metric("Exactitud", num((tp + tn) / len(y), 2), help="Aciertos totales. Engaña con clases desbalanceadas.")
    pers = d["Regla: persistencia (repetir el mes anterior) · clase"].to_numpy()
    siempre = f1_score(y, np.ones_like(y))
    st.caption(f"Para comparar, en estos mismos datos: la persistencia da F1 {num(f1_score(y, pers), 2)} y decir "
               f"siempre «sube», sin aprender nada, da {num(siempre, 2)}. Hay {int(y.sum())} «sube» y {int(len(y) - y.sum())} «baja».")
    if u < 0.5 and (pred.mean() > 0.7):
        st.warning(f"Con este umbral el modelo dice «sube» en el {num(pred.mean() * 100, 0)} % de los casos. Si el F1 sube, "
                   "es porque se parece a «decir siempre sube», no porque ordene mejor.")

caja("**Cómo se mueve:** si bajás el umbral, el modelo dice «sube» más seguido: detecta más subas (sube la "
     "exhaustividad) pero inventa más (baja la precisión). Si lo subís, pasa lo contrario. El umbral se elige según "
     "qué error cuesta más en el problema real.")

# ------------------------------------------------------------- curvas
fpr, tpr, thr_r = roc_curve(y, p)
pr, rc, thr_p = precision_recall_curve(y, p)
fpr_u = fp / max(fp + tn, 1); tpr_u = rec
c1, c2 = st.columns(2)
with c1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=VERDE, width=3), name=f"{m}"))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dash"), name="Azar (AUC 0,5)"))
    f.add_trace(go.Scatter(x=[fpr_u], y=[tpr_u], mode="markers", marker=dict(size=14, color=NARANJA, line=dict(color=TINTA, width=1)),
                           name=f"Umbral {num(u, 2)}"))
    f.update_xaxes(title="Falsas alarmas (bajó y dijo «sube»)", range=[0, 1]); f.update_yaxes(title="Subas detectadas", range=[0, 1.02])
    st.plotly_chart(layout(f, 400, f"Curva ROC · AUC {num(auc(fpr, tpr), 2)}"), use_container_width=True)
with c2:
    f = go.Figure()
    f.add_trace(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=VERDE, width=3), name=f"{m}"))
    f.add_hline(y=y.mean(), line_dash="dash", line_color=GRIS, annotation_text="azar", annotation_position="bottom right")
    f.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", marker=dict(size=14, color=NARANJA, line=dict(color=TINTA, width=1)),
                           name=f"Umbral {num(u, 2)}"))
    f.update_xaxes(title="Exhaustividad (subas detectadas)", range=[0, 1.02]); f.update_yaxes(title="Precisión", range=[0, 1.02])
    st.plotly_chart(layout(f, 400, f"Curva precisión-exhaustividad · PM {num(average_precision_score(y, p), 2)}"), use_container_width=True)

au = auc(fpr, tpr)
if au < 0.5:
    caja(f"**AUC {num(au, 2)}, menor que 0,5:** la curva queda por debajo de la diagonal del azar. El modelo ordena al "
         "revés: les da más probabilidad de «sube» a los meses que bajaron. Aprendió en la caída y se lo evaluó en la "
         "recuperación. Por eso **ningún umbral lo arregla**: mover el umbral cambia cuántos «sube» dice, no el orden.", "aviso")

# ------------------------------------------------------------- F1 según el umbral
with st.expander("¿Qué umbral da el mejor F1?"):
    us = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y, (p >= v).astype(int), zero_division=0) for v in us]
    g = go.Figure(go.Scatter(x=us, y=f1s, mode="lines", line=dict(color=VERDE, width=3)))
    g.add_vline(x=u, line_color=NARANJA, annotation_text="tu umbral")
    g.add_hline(y=f1_score(y, pers), line_dash="dash", line_color=GRIS, annotation_text="persistencia")
    g.update_xaxes(title="Umbral"); g.update_yaxes(title="F1 «sube»", range=[0, 1])
    st.plotly_chart(layout(g, 320, leyenda_abajo=False), use_container_width=True)
    st.caption("Elegir el umbral mirando la prueba sería hacer trampa: en el trabajo usamos 0,5. Este gráfico es para "
               "entender el efecto, no para elegir.")
