"""
Tablero del proyecto CAMMESA · Aprendizaje Automático y Grandes Datos (UNRaf, 2026)
=====================================================================================
Punto de entrada. Cada sección es una página de la carpeta dashboard/views/.

    Inicio                      qué resuelve el proyecto y cómo leer el tablero
    TP2 · Regresión             probar el pronóstico, comparar modelos, jugar con las variables
    TP3 · Clasificación         probar «sube o baja», mover el umbral, comparar modelos
    Agrupamiento                qué representa cada grupo y en qué grupo está cada región o agente
    Preguntas frecuentes        respuestas cortas a las preguntas de la cátedra

Ejecutar localmente:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Demanda eléctrica regional · CAMMESA", page_icon="⚡", layout="wide")

paginas = {
    "": [st.Page("dashboard/views/inicio.py", title="Inicio", icon="🏠", default=True)],
    "TP2 · Regresión (¿cuántos GWh?)": [
        st.Page("dashboard/views/tp2_probar.py", title="Probar el pronóstico", icon="📈"),
        st.Page("dashboard/views/tp2_comparar.py", title="Comparar modelos", icon="📊"),
        st.Page("dashboard/views/tp2_variables.py", title="Jugar con las variables", icon="🎛️"),
    ],
    "TP3 · Clasificación (¿sube o baja?)": [
        st.Page("dashboard/views/tp3_probar.py", title="Probar «sube o baja»", icon="🔮"),
        st.Page("dashboard/views/tp3_umbral.py", title="Umbral y curvas", icon="🎚️"),
        st.Page("dashboard/views/tp3_comparar.py", title="Comparar modelos", icon="🧮"),
    ],
    "Agrupamiento": [
        st.Page("dashboard/views/grupos.py", title="Grupos y buscador", icon="🧩"),
    ],
    "Ayuda": [
        st.Page("dashboard/views/preguntas.py", title="Preguntas frecuentes", icon="❓"),
    ],
}
st.navigation(paginas).run()

st.sidebar.caption("Grupo 6 · Alemandi, Kirov, Palavecino · UNRaf 2026. Datos: CAMMESA, «Demanda últimos años».")
