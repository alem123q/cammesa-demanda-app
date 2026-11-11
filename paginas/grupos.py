"""Agrupamiento: qué representa cada grupo y en qué grupo está cada región o cada agente (cliente)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from comun import MESES, MESES_CORTOS, NOMBRES, PALETA, TINTA, VERDE, caja, csv, jsn, layout, num, pct, periodo_txt

st.title("🧩 Agrupamiento")
st.caption("Sin decirle la respuesta, el algoritmo k-medias junta los casos que se parecen. "
           "¿Los grupos que encuentra coinciden con algo que entendamos?")

tab_r, tab_a = st.tabs(["🗺️ Regiones y meses (TP3)", "🏭 Agentes: distribuidoras y grandes usuarios"])
LETRA = "ABCDEFGH"


def curva_silueta(curva, k_sel, titulo):
    c = pd.DataFrame(curva)
    f = go.Figure(go.Scatter(x=c["k"], y=c["silueta"], mode="lines+markers+text", line=dict(color=VERDE, width=3),
                             text=[num(v, 2) for v in c["silueta"]], textposition="top center", marker=dict(size=9)))
    f.add_vline(x=k_sel, line_color="#E89B25", annotation_text=f"k = {k_sel}")
    f.update_xaxes(title="Cantidad de grupos (k)", dtick=1)
    f.update_yaxes(title="Silueta (más es mejor)", range=[min(c["silueta"]) - 0.03, max(c["silueta"]) + 0.05])
    return layout(f, 300, titulo, leyenda_abajo=False)


# =================================================================== regiones y meses
with tab_r:
    g, curva = csv("tp3_grupos.csv"), jsn("tp3_grupos_curva.json")
    mejor = int(pd.DataFrame(curva).set_index("k")["silueta"].idxmax())
    c1, c2 = st.columns([1, 1.6])
    with c1:
        k = st.slider("Cantidad de grupos (k)", 2, 8, mejor, key="k_reg",
                      help=f"La silueta mide qué tan separados quedan los grupos. Es máxima con k = {mejor}.")
        fila = next(c for c in curva if c["k"] == k)
        st.metric("Silueta", num(fila["silueta"], 3))
        st.metric("Acuerdo con «sube o baja» (ARI)", num(fila["ari_sube"], 2),
                  help="Índice de Rand ajustado: 1 = los grupos coinciden con la clase; 0 = nada que ver.")
    with c2:
        st.plotly_chart(curva_silueta(curva, k, "¿Cuántos grupos conviene armar?"), use_container_width=True)

    col = f"grupo_k{k}"
    ent = g[g["conjunto"] == "entrenamiento"]
    st.subheader("Qué representa cada grupo")

    def describir(d):
        gba = (d["region"] == "GRAN BS.AS.").mean()
        meses = sorted(d["mes"].unique())
        cuenta = ent.groupby("mes").size()
        propios = [m for m in meses if (d["mes"] == m).sum() >= 0.5 * cuenta.get(m, 1)]
        if gba >= 0.8:
            nombre = "Gran Buenos Aires"
        elif gba == 0 and propios and len(propios) < 12:
            nombre = f"Resto de las regiones, {MESES_CORTOS[propios[0] - 1]} a {MESES_CORTOS[propios[-1] - 1]}" \
                if propios == list(range(propios[0], propios[-1] + 1)) else "Resto de las regiones, " + ", ".join(MESES_CORTOS[m - 1] for m in propios)
        else:
            nombre = "Mezcla de regiones y meses"
        return nombre, gba, propios

    desc = {}
    cols = st.columns(min(k, 4))
    for i, (gr, d) in enumerate(ent.groupby(col)):
        nombre, gba, propios = describir(d)
        desc[gr] = nombre
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**Grupo {LETRA[gr]}** · {len(d)} casos")
            st.markdown(f"### {nombre}")
            regs = d["region"].value_counts()
            st.caption(f"Regiones: {', '.join(NOMBRES[r] for r in regs.index[:4])}{'…' if len(regs) > 4 else ''}. "
                       f"Meses: {', '.join(MESES_CORTOS[m - 1] for m in sorted(d['mes'].unique()))}. "
                       f"«Sube» en el {num(d['sube'].mean() * 100, 0)} % de los casos.")
    caja(f"Los grupos separan **tamaño** (Gran Buenos Aires) y **época del año**, no tendencia: su acuerdo con «sube o "
         f"baja» es {num(fila['ari_sube'], 2)}. Con estas variables, lo que separa «sube» de «baja» no se ve.")

    st.subheader("Mapa de los grupos")
    color_por = st.radio("Colorear por", ["Grupo", "Lo que pasó (sube o baja)", "Región"], horizontal=True)
    d = ent.assign(Grupo=ent[col].map(lambda x: f"{LETRA[x]}: {desc[x]}"),
                   **{"Lo que pasó (sube o baja)": ent["sube"].map({1: "sube", 0: "baja"}),
                      "Región": ent["region"].map(NOMBRES), "Mes": ent["indice_tiempo"].map(periodo_txt)})
    fig = px.scatter(d, x="pc1", y="pc2", color=color_por, hover_data={"Región": True, "Mes": True, "pc1": False, "pc2": False},
                     color_discrete_sequence=PALETA)
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.update_xaxes(title="Componente principal 1"); fig.update_yaxes(title="Componente principal 2")
    st.plotly_chart(layout(fig, 450), use_container_width=True)
    st.caption("Cada punto es una región en un mes del entrenamiento. Las 13 variables se resumen en 2 ejes (PCA) solo "
               "para poder dibujarlas. Probá colorear por «lo que pasó»: los «sube» y «baja» están mezclados en todos los grupos.")

    st.subheader("🔎 ¿En qué grupo está mi región?")
    c1, c2 = st.columns(2)
    reg = c1.selectbox("Región", list(NOMBRES), format_func=NOMBRES.get, key="b_reg")
    pers = sorted(g["indice_tiempo"].unique())
    per = c2.selectbox("Mes", pers, index=pers.index("2019-10"), format_func=periodo_txt, key="b_per")
    f = g[(g["region"] == reg) & (g["indice_tiempo"] == per)].iloc[0]
    st.success(f"**{NOMBRES[reg]}, {periodo_txt(per)}** → **Grupo {LETRA[int(f[col])]}: {desc[int(f[col])]}**. "
               f"Ese mes la demanda {'subió' if f['sube'] else 'bajó'} frente al año anterior."
               + (" (Mes de la prueba: el grupo se asigna con los centros aprendidos en el entrenamiento.)" if f["conjunto"] == "prueba" else ""))


# =================================================================== agentes
with tab_a:
    a, curvas = csv("agentes_perfiles.csv"), jsn("agentes_curvas.json")
    st.markdown("Cada **agente** es un cliente del mercado eléctrico: una distribuidora (como EDENOR o una cooperativa) "
                "o un gran usuario (una fábrica, un shopping, un aeropuerto). Resumimos la historia de cada uno, por tarifa, "
                f"en seis números y los agrupamos. Son **{len(a)} series** con al menos 24 meses de datos.")
    VAR = {"log_tamano": "Tamaño", "estacionalidad": "Estacionalidad", "volatilidad": "Volatilidad",
           "tendencia_anual": "Tendencia", "frac_caidas": "Caídas frecuentes", "cv": "Variabilidad"}
    c1, c2 = st.columns([1, 1.6])
    with c1:
        con = st.toggle("Incluir «caídas frecuentes»", value=False,
                        help="La fracción de meses con caídas de más del 15 %. Se parece mucho a la volatilidad "
                             "(correlación 0,77): es casi la misma información dos veces.")
        clave = "con" if con else "sin"
        cv_ = curvas[clave]
        mejor = int(pd.DataFrame(cv_).set_index("k")["silueta"].idxmax())
        ka = st.slider("Cantidad de grupos (k)", 2, 8, 3, key="k_ag")
        st.metric("Silueta", num(next(c for c in cv_ if c["k"] == ka)["silueta"], 3))
    with c2:
        st.plotly_chart(curva_silueta(cv_, ka, "¿Cuántos grupos conviene armar?"), use_container_width=True)
    if not con:
        caja("Sin «caídas frecuentes» la silueta sube (de 0,30 a 0,44 con k = 3): sacar una variable redundante, que "
             "repetía la volatilidad, hace los grupos más claros. Es un ejemplo de variable que hace ruido.", "ok")

    cols_vars = [v for v in VAR if con or v != "frac_caidas"]
    colg = f"grupo_{clave}_k{ka}"
    z = (a[cols_vars] - a[cols_vars].mean()) / a[cols_vars].std()
    TXT = {"log_tamano": ("grandes", "chicos"), "estacionalidad": ("consumen más en verano", "consumen más en invierno"),
           "volatilidad": ("volátiles", "estables mes a mes"), "tendencia_anual": ("en crecimiento", "en caída"),
           "frac_caidas": ("con caídas frecuentes", "casi sin caídas"), "cv": ("muy variables", "parejos")}

    def nombre_grupo(gr):
        m = z[a[colg] == gr].mean()
        partes, usados = [], set()
        for v in m.abs().sort_values(ascending=False).index:          # volatilidad y variabilidad cuentan como una idea
            idea = "var" if v in ("volatilidad", "cv") else v
            if abs(m[v]) > 0.3 and idea not in usados and len(partes) < 2:
                partes.append(TXT[v][0] if m[v] > 0 else TXT[v][1]); usados.add(idea)
        return (", ".join(partes) or "promedio").capitalize()

    nombres_g = {gr: nombre_grupo(gr) for gr in sorted(a[colg].unique())}

    def perfil(d):
        return {"Agentes": len(d), "Tamaño típico (MWh/mes)": 10 ** d["log_tamano"].median(),
                "Verano / invierno": 2 ** d["estacionalidad"].median(),
                "Cambio mensual típico (%)": d["volatilidad"].median() * 100,
                "Tendencia (% por año)": (10 ** d["tendencia_anual"].median() - 1) * 100,
                "Meses con caída > 15 % (%)": d["frac_caidas"].median() * 100,
                "Grandes usuarios (%)": (d["categoria_demanda"] == "Gran Usuario").mean() * 100}

    st.subheader("Qué representa cada grupo")
    cols = st.columns(min(ka, 4))
    for i, gr in enumerate(sorted(nombres_g)):
        d = a[a[colg] == gr]; p = perfil(d)
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**Grupo {LETRA[gr]}** · {len(d)} series")
            st.markdown(f"### {nombres_g[gr]}")
            st.caption(f"Tamaño típico {num(p['Tamaño típico (MWh/mes)'], 0)} MWh/mes · verano/invierno ×{num(p['Verano / invierno'], 2)} · "
                       f"cambio mensual {num(p['Cambio mensual típico (%)'], 0)} % · tendencia {pct(p['Tendencia (% por año)'], dec=0)} por año · "
                       f"{num(p['Grandes usuarios (%)'], 0)} % grandes usuarios.")
    tabla = pd.DataFrame({f"{LETRA[gr]}: {nombres_g[gr]}": perfil(a[a[colg] == gr]) for gr in sorted(nombres_g)}).T
    with st.expander("Ver la tabla de perfiles"):
        st.dataframe(tabla.style.format("{:,.1f}"), use_container_width=True)
        st.caption("Medianas de cada grupo. Verano/invierno: 1,5 significa que consume 50 % más en verano. "
                   "Los grupos casi no coinciden con la tarifa ni con el tipo de agente (ARI ≈ 0,07).")

    st.subheader("🔎 ¿En qué grupo está mi cliente?")
    a["etiqueta"] = a["agente"].fillna(a["agente_nemo"]) + " · " + a["categoria_tarifa"] + " (" + a["region"].map(NOMBRES) + ")"
    opciones = a.sort_values("etiqueta")["etiqueta"].tolist()
    defecto = next((i for i, o in enumerate(opciones) if o.startswith("EDENOR")), 0)
    sel = st.selectbox("Buscá un agente (escribí parte del nombre)", opciones, index=defecto)
    f = a[a["etiqueta"] == sel].iloc[0]
    gr = int(f[colg])
    st.success(f"**{sel}** → **Grupo {LETRA[gr]}: {nombres_g[gr]}**")
    comp = pd.DataFrame({"Este agente": perfil(a[a["etiqueta"] == sel]), f"Grupo {LETRA[gr]} (mediana)": perfil(a[a[colg] == gr]),
                         "Todos (mediana)": perfil(a)}).drop(["Agentes", "Grandes usuarios (%)"])
    st.dataframe(comp.style.format("{:,.1f}"), use_container_width=True)
    with st.expander(f"Otros agentes del grupo {LETRA[gr]}"):
        otros = a[a[colg] == gr].assign(tam=lambda x: 10 ** x["log_tamano"]).sort_values("tam", ascending=False)
        st.dataframe(otros[["etiqueta", "categoria_demanda", "tam"]].rename(columns={"etiqueta": "Agente", "categoria_demanda": "Tipo",
                                                                                      "tam": "Tamaño (MWh/mes)"})
                     .style.format({"Tamaño (MWh/mes)": "{:,.0f}"}), hide_index=True, use_container_width=True, height=300)
