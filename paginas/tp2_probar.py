"""
TP2 · Probar el pronóstico (página del tablero)
===================================================
Aplicación Streamlit del TP4 (Aprendizaje Automático y Grandes Datos · UNRaf).
Despliega el modelo pre-registrado del TP2: Ridge (alpha = 3,16), siempre mostrado
junto a la regla simple «mismo mes del año anterior», que en la prueba no pudo superar.

Ejecutar localmente:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "modelo" / "ridge_regional.joblib"
RUTA_META = RAIZ / "modelo" / "metadatos.json"
RUTA_DATOS = RAIZ / "datos" / "serie_regional.csv"
RUTA_REGISTRO = RAIZ / "registro" / "predicciones.csv"

COLUMNAS = ["region", "rezago_1", "rezago_12", "mes", "trimestre"]
NOMBRES = {"BUENOS AIRES": "Buenos Aires (interior)", "CENTRO": "Centro", "COMAHUE": "Comahue", "CUYO": "Cuyo",
           "GRAN BS.AS.": "Gran Buenos Aires", "LITORAL": "Litoral", "NORESTE": "Noreste",
           "NOROESTE": "Noroeste", "PATAGONICA": "Patagónica"}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
ANIOS = list(range(2018, 2022))          # 2021 como máximo: más allá el modelo extrapola demasiado
NARANJA, VERDE, GRIS, TINTA = "#E89B25", "#2A7F86", "#9AA3AE", "#1C2433"


# ---------------------------------------------------------------- carga (una vez por sesión del servidor)
@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_meta() -> dict:
    return json.loads(RUTA_META.read_text(encoding="utf-8"))


@st.cache_data
def cargar_serie() -> pd.DataFrame:
    s = pd.read_csv(RUTA_DATOS, dtype={"indice_tiempo": str})
    s["periodo"] = pd.PeriodIndex(s["indice_tiempo"], freq="M")
    return s


def gwh(mwh: float) -> str:
    return f"{mwh / 1000:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") + " GWh"


def pct(x: float, signo: bool = True) -> str:
    return (f"{x:+.1f}" if signo else f"{x:.1f}").replace(".", ",") + " %"


def valor_historico(serie: pd.DataFrame, region: str, periodo: pd.Period) -> float | None:
    fila = serie[(serie["region"] == region) & (serie["periodo"] == periodo)]
    return float(fila["demanda_MWh"].iloc[0]) if len(fila) else None


def predecir(modelo, region: str, mes: int, r1_mwh: float, r12_mwh: float) -> float:
    X = pd.DataFrame([{"region": region, "rezago_1": r1_mwh, "rezago_12": r12_mwh,
                       "mes": mes, "trimestre": (mes - 1) // 3 + 1}])[COLUMNAS]
    return float(modelo.predict(X)[0])


def validar(region: str, r1: float, r12: float, rangos: dict) -> tuple[list[str], list[str]]:
    """Devuelve (errores que impiden predecir, advertencias). Valores en MWh."""
    errores, avisos = [], []
    lo, hi = rangos[region]["min_MWh"], rangos[region]["max_MWh"]
    for nombre, v in [("mes anterior", r1), ("mismo mes del año anterior", r12)]:
        if v is None or not np.isfinite(v) or v <= 0:
            errores.append(f"La demanda del {nombre} tiene que ser un número mayor que cero.")
        elif v < 0.3 * lo or v > 3 * hi:
            errores.append(f"La demanda del {nombre} ({gwh(v)}) está muy lejos de lo que consumió esta región "
                           f"({gwh(lo)} a {gwh(hi)}). ¿Puede ser un error de unidades? Se piden GWh.")
        elif v < 0.8 * lo or v > 1.2 * hi:
            avisos.append(f"La demanda del {nombre} ({gwh(v)}) sale del rango histórico de la región "
                          f"({gwh(lo)} a {gwh(hi)}): el modelo va a extrapolar.")
    if not errores:
        ratio = r1 / r12
        if ratio < 0.6 or ratio > 1.6:
            avisos.append(f"El mes anterior es {ratio:.2f} veces el mismo mes del año anterior: es un salto poco "
                          "habitual (puede ser estacional, pero revisá los valores).")
    return errores, avisos


def registrar(fila: dict) -> None:
    """Agrega la predicción al registro (monitoreo básico de uso y de posible deriva)."""
    try:
        RUTA_REGISTRO.parent.mkdir(exist_ok=True)
        pd.DataFrame([fila]).to_csv(RUTA_REGISTRO, mode="a", header=not RUTA_REGISTRO.exists(), index=False)
    except OSError:
        pass                                    # en un servidor de solo lectura, se sigue sin registrar en disco
    st.session_state.setdefault("registro", []).append(fila)


def contribuciones(modelo, serie: pd.DataFrame, region: str, mes: int, r1: float, r12: float) -> tuple[float, pd.Series]:
    """Explica el pronóstico frente a un mes «típico» de la misma región.

    Ridge trabaja en logaritmos: pronóstico = exp(intercepto + Σ coef·z). Por eso cada predictora
    multiplica el pronóstico por un factor. La referencia es la misma región con sus demandas
    promedio del entrenamiento y un mes promedio; cada barra es el % que agrega o quita esa variable.
    """
    def z(X):
        m = modelo.regressor_.named_steps["prep"].transform(X[COLUMNAS])
        return m.toarray()[0] if hasattr(m, "toarray") else np.asarray(m)[0]

    ent = serie[(serie["region"] == region) & (serie["indice_tiempo"] >= "2018-01") & (serie["indice_tiempo"] < "2019-09")]
    tipico = float(np.expm1(np.log1p(ent["demanda_MWh"]).mean()))
    X = pd.DataFrame([{"region": region, "rezago_1": r1, "rezago_12": r12, "mes": mes, "trimestre": (mes - 1) // 3 + 1}])
    X_ref = X.assign(rezago_1=tipico, rezago_12=tipico)
    nombres = modelo.regressor_.named_steps["prep"].get_feature_names_out()
    zx, zr = z(X), z(X_ref)
    zr[np.char.startswith(nombres.astype(str), "calendario__")] = 0.0      # mes promedio
    aporte = pd.Series((zx - zr) * modelo.regressor_.named_steps["modelo"].coef_, index=nombres)
    grupos = pd.Series({"Demanda del mes anterior": aporte["rezagos__rezago_1"],
                        "Demanda del mismo mes del año anterior": aporte["rezagos__rezago_12"],
                        "Mes del año": aporte[aporte.index.str.startswith("calendario__")].sum()})
    base_ref = float(modelo.predict(X_ref[COLUMNAS])[0] / np.exp(aporte[aporte.index.str.startswith("calendario__")].sum()))
    return base_ref, (np.exp(grupos) - 1) * 100


# ---------------------------------------------------------------- página

try:
    modelo, meta, serie = cargar_modelo(), cargar_meta(), cargar_serie()
except Exception as e:                          # sin modelo no hay aplicación: mensaje claro en vez de una traza
    st.error(f"No se pudo cargar el modelo o los datos ({e}). Ejecutá `python entrenar.py` y volvé a abrir la app.")
    st.stop()

mr, mb = meta["prueba_ridge"], meta["prueba_base"]
q = meta["error_pct_cuantiles"]

st.title("📈 TP2 · Probar el pronóstico")
st.caption("¿Cuánta energía va a consumir cada región el mes que viene?")
st.markdown(
    "Pronóstico de la **demanda eléctrica mensual** de las 9 regiones del mercado eléctrico argentino, "
    "con datos públicos de **CAMMESA** (enero 2017 – febrero 2020). El modelo es una **regresión Ridge** "
    "entrenada con la demanda del mes anterior y la del mismo mes del año anterior. "
    "Al lado siempre se muestra la regla más simple, **copiar el mismo mes del año anterior**: en la "
    "evaluación, el modelo no logró superarla, y preferimos que lo veas.")

tab1, tab2, tab4 = st.tabs(["📈 Un pronóstico", "📄 Muchos a la vez (CSV)", "🗒️ Registro"])

# ================================================================ 1 · pronóstico individual
with tab1:
    c1, c2, c3 = st.columns([1.4, 1, 1])
    region = c1.selectbox("Región", list(NOMBRES), format_func=NOMBRES.get, index=4)
    anio = c2.selectbox("Año a pronosticar", ANIOS, index=ANIOS.index(2020))
    mes = c3.selectbox("Mes a pronosticar", range(1, 13), format_func=lambda m: MESES[m - 1].capitalize(), index=2)

    objetivo = pd.Period(year=anio, month=mes, freq="M")
    p1, p12 = objetivo - 1, objetivo - 12
    h1, h12 = valor_historico(serie, region, p1), valor_historico(serie, region, p12)
    real = valor_historico(serie, region, objetivo)

    st.caption("Las demandas se completan solas con la historia de CAMMESA cuando existe; podés cambiarlas "
               "para simular otro escenario. Unidades: GWh (1 GWh = 1.000 MWh).")
    c4, c5 = st.columns(2)
    ult = serie[serie["region"] == region].iloc[-1]
    r1_gwh = c4.number_input(f"Demanda del mes anterior ({MESES[p1.month - 1]} {p1.year}), GWh",
                             min_value=0.0, max_value=20000.0, step=10.0, format="%.1f",
                             value=round((h1 if h1 else ult["demanda_MWh"]) / 1000, 1),
                             key=f"r1_{region}_{objetivo}")
    r12_gwh = c5.number_input(f"Demanda del mismo mes del año anterior ({MESES[p12.month - 1]} {p12.year}), GWh",
                              min_value=0.0, max_value=20000.0, step=10.0, format="%.1f",
                              value=round((h12 if h12 else ult["demanda_MWh"]) / 1000, 1),
                              key=f"r12_{region}_{objetivo}")
    if h1 is None or h12 is None:
        st.info("Para este mes no hay datos históricos de alguno de los dos valores: se cargó el último mes "
                "disponible como punto de partida. Ingresá los valores reales si los conocés.")
    if objetivo > pd.Period("2020-03", freq="M"):
        st.warning("Pronosticás fuera del período con datos (hasta febrero de 2020). El modelo no conoce lo que "
                   "pasó después, como la pandemia de 2020: tomalo como un ejercicio.")

    errores, avisos = validar(region, r1_gwh * 1000, r12_gwh * 1000, meta["rangos"])
    for a in avisos:
        st.warning(a)
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    pred = predecir(modelo, region, mes, r1_gwh * 1000, r12_gwh * 1000)
    base = r12_gwh * 1000
    lo, hi = pred * (1 + q["10"] / 100), pred * (1 + q["90"] / 100)

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Pronóstico del modelo · {MESES[mes - 1]} {anio}", gwh(pred),
              delta=pct((pred / base - 1) * 100) + " frente al año anterior", delta_color="off")
    k2.metric("Regla simple: igual que hace un año", gwh(base))
    if real is not None:
        k3.metric("Lo que realmente se consumió", gwh(real),
                  delta=f"error del modelo {pct((pred / real - 1) * 100)}", delta_color="off")
    else:
        k3.metric("Rango esperable, GWh (8 de cada 10 casos)",
                  f"{lo / 1000:,.0f} – {hi / 1000:,.0f}".replace(",", "."))
    st.markdown(
        f"En la evaluación, el valor real quedó entre **{pct(q['10'])}** y **{pct(q['90'])}** del pronóstico en "
        f"8 de cada 10 casos. El error típico en esta región fue de "
        f"**{meta['por_region'][region]['Ridge']['RMSE_GWh']:.0f} GWh** con el modelo y "
        f"**{meta['por_region'][region]['Base']['RMSE_GWh']:.0f} GWh** con la regla simple.")

    # --- gráfico: historia de la región + pronóstico
    hist = serie[serie["region"] == region]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["periodo"].dt.to_timestamp(), y=hist["demanda_MWh"] / 1000, mode="lines+markers",
                             name="Historia (CAMMESA)", line=dict(color=GRIS, width=2), marker=dict(size=4),
                             hovertemplate="%{x|%b %Y}: %{y:,.0f} GWh<extra></extra>"))
    t = objetivo.to_timestamp()
    fig.add_trace(go.Scatter(x=[t, t], y=[lo / 1000, hi / 1000], mode="lines", name="Rango esperable",
                             line=dict(color=VERDE, width=10), opacity=0.25, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[t], y=[base / 1000], mode="markers", name="Regla simple",
                             marker=dict(color=NARANJA, size=12, symbol="diamond"),
                             hovertemplate="Regla simple: %{y:,.0f} GWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t], y=[pred / 1000], mode="markers", name="Pronóstico del modelo",
                             marker=dict(color=VERDE, size=14),
                             hovertemplate="Modelo: %{y:,.0f} GWh<extra></extra>"))
    if real is not None:
        fig.add_trace(go.Scatter(x=[t], y=[real / 1000], mode="markers", name="Valor real",
                                 marker=dict(color=TINTA, size=11, symbol="x"),
                                 hovertemplate="Real: %{y:,.0f} GWh<extra></extra>"))
    fig.add_vrect(x0=pd.Timestamp("2019-09-01"), x1=pd.Timestamp("2020-02-28"), fillcolor=NARANJA, opacity=0.07,
                  line_width=0, annotation_text="período de prueba", annotation_position="top left")
    fig.update_layout(title=f"Demanda mensual · {NOMBRES[region]}", yaxis_title="GWh", height=420,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15),
                      plot_bgcolor="white", hovermode="x unified")
    fig.update_yaxes(gridcolor="#E3E7EB", separatethousands=True)
    fig.update_xaxes(tickformat="%m/%Y")
    st.plotly_chart(fig, use_container_width=True)

    # --- explicabilidad
    with st.expander("¿Por qué el modelo dio este número?"):
        ref, ap = contribuciones(modelo, serie, region, mes, r1_gwh * 1000, r12_gwh * 1000)
        st.markdown(f"Para un mes típico de {NOMBRES[region]} (demandas promedio de 2018–2019), el modelo daría "
                    f"**{gwh(ref)}**. Las barras muestran cuánto mueve ese punto de partida cada dato que ingresaste:")
        fe = go.Figure(go.Bar(x=ap.values, y=ap.index, orientation="h",
                              marker_color=[VERDE if v >= 0 else NARANJA for v in ap.values],
                              text=[pct(v) for v in ap.values], textposition="outside"))
        fe.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="white",
                         xaxis_title="Efecto sobre el pronóstico (%)")
        st.plotly_chart(fe, use_container_width=True)
        st.caption("Los efectos se multiplican: punto de partida × (1 + cada efecto) da el pronóstico. Las dos demandas "
                   "pasadas son las que más pesan: el modelo, en esencia, combina el mes anterior con el mismo mes del "
                   "año anterior y casi no agrega nada propio. Por eso se parece tanto a la regla simple.")

    if st.button("Guardar esta predicción en el registro"):
        registrar({"momento": datetime.now().isoformat(timespec="seconds"), "origen": "individual",
                   "region": region, "periodo": str(objetivo), "mes_anterior_GWh": r1_gwh,
                   "mismo_mes_anio_anterior_GWh": r12_gwh, "pronostico_GWh": round(pred / 1000, 2),
                   "regla_simple_GWh": round(base / 1000, 2),
                   "real_GWh": round(real / 1000, 2) if real is not None else None})
        st.success("Guardada. La ves en la pestaña «Registro».")

# ================================================================ 2 · lote
with tab2:
    st.markdown("Subí un CSV con una fila por pronóstico y estas columnas: `region`, `anio`, `mes`, "
                "`mes_anterior_GWh`, `mismo_mes_anio_anterior_GWh`. La región puede escribirse como en CAMMESA "
                "(por ejemplo `GRAN BS.AS.`) o con el nombre de la lista (`Gran Buenos Aires`).")
    ejemplo = pd.DataFrame({"region": ["GRAN BS.AS.", "Patagónica", "CENTRO"], "anio": [2020, 2020, 2020],
                            "mes": [3, 3, 3], "mes_anterior_GWh": [4300.0, 520.0, 1050.0],
                            "mismo_mes_anio_anterior_GWh": [4500.0, 560.0, 1080.0]})
    st.download_button("Descargar plantilla de ejemplo", ejemplo.to_csv(index=False).encode("utf-8"),
                       "plantilla_pronostico.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo como CSV: {e}")
            st.stop()
        faltan = {"region", "anio", "mes", "mes_anterior_GWh", "mismo_mes_anio_anterior_GWh"} - set(lote.columns)
        if faltan:
            st.error(f"Faltan columnas: {', '.join(sorted(faltan))}.")
            st.stop()
        if len(lote) > 5000:
            st.error("El archivo tiene más de 5.000 filas; dividilo en partes.")
            st.stop()
        inversos = {v.lower(): k for k, v in NOMBRES.items()} | {k.lower(): k for k in NOMBRES}
        salida = []
        for _, f in lote.iterrows():
            reg = inversos.get(str(f["region"]).strip().lower())
            obs, pr = "", np.nan
            try:
                m_, a_ = int(f["mes"]), int(f["anio"])
                r1, r12 = float(f["mes_anterior_GWh"]) * 1000, float(f["mismo_mes_anio_anterior_GWh"]) * 1000
            except (TypeError, ValueError):
                m_ = a_ = None
            if reg is None:
                obs = "región desconocida"
            elif m_ is None or not 1 <= m_ <= 12 or a_ not in ANIOS:
                obs = "año, mes o demandas no válidos"
            else:
                err, av = validar(reg, r1, r12, meta["rangos"])
                if err:
                    obs = "fuera de rango (¿unidades?)"
                else:
                    pr = predecir(modelo, reg, m_, r1, r12) / 1000
                    obs = "extrapola" if av else "ok"
                    registrar({"momento": datetime.now().isoformat(timespec="seconds"), "origen": "lote",
                               "region": reg, "periodo": f"{a_}-{m_:02d}", "mes_anterior_GWh": r1 / 1000,
                               "mismo_mes_anio_anterior_GWh": r12 / 1000, "pronostico_GWh": round(pr, 2),
                               "regla_simple_GWh": round(r12 / 1000, 2), "real_GWh": None})
            salida.append({**f.to_dict(), "pronostico_modelo_GWh": round(pr, 1) if np.isfinite(pr) else None,
                           "regla_simple_GWh": f["mismo_mes_anio_anterior_GWh"], "estado": obs})
        res = pd.DataFrame(salida)
        ok = res["estado"].isin(["ok", "extrapola"]).sum()
        if ok == len(res):
            st.success(f"{ok} de {len(res)} filas pronosticadas.")
        else:
            st.warning(f"{ok} de {len(res)} filas pronosticadas; revisá la columna «estado».")
        st.dataframe(res, use_container_width=True, hide_index=True,
                     column_config={"anio": st.column_config.NumberColumn(format="%d"),
                                    "mes": st.column_config.NumberColumn(format="%d")})
        st.download_button("Descargar resultados", res.to_csv(index=False).encode("utf-8"),
                           "pronosticos.csv", "text/csv")

# ================================================================ 4 · registro
with tab4:
    st.markdown("Cada predicción guardada queda registrada con su fecha y sus entradas. Sirve para analizar el uso "
                "y detectar **deriva**: si con el tiempo el error contra los valores reales crece, hay que reentrenar.")
    reg_ses = pd.DataFrame(st.session_state.get("registro", []))
    if reg_ses.empty:
        st.info("Todavía no guardaste predicciones en esta sesión.")
    else:
        st.dataframe(reg_ses, use_container_width=True, hide_index=True)
        st.download_button("Descargar registro", reg_ses.to_csv(index=False).encode("utf-8"),
                           "registro_predicciones.csv", "text/csv")
        con_real = reg_ses.dropna(subset=["real_GWh"])
        if len(con_real):
            e = (con_real["pronostico_GWh"] / con_real["real_GWh"] - 1) * 100
            st.metric("Error porcentual medio de lo registrado con valor real", pct(e.abs().mean(), signo=False))

