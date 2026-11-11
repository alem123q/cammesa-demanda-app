"""Preguntas frecuentes: respuestas cortas a las preguntas de la cátedra, con enlace a la página donde se ve."""

import streamlit as st

st.title("❓ Preguntas frecuentes")
st.caption("Respuestas cortas, y dónde verlo en el tablero.")

P = [
    ("¿Qué variables usan y por qué tantas?",
     "Son cinco ideas, aunque ocupen 12 columnas (la región se abre en una columna por región): **región**, **demanda del "
     "mes anterior**, **demanda del mismo mes del año anterior**, **mes** y **trimestre**. En el TP3 se suma la **variación "
     "interanual del mes anterior**. Cada una sale del análisis del TP1: la región fija el nivel, el año anterior captura "
     "la estacionalidad propia de cada región y el mes anterior la persistencia. Excluimos tarifa y categoría porque en "
     "febrero de 2019 cambió el registro.",
     "TP2 · Jugar con las variables"),
    ("Si una variable o categoría hace ruido, ¿qué hacen?",
     "Se prueba sacarla y se mira el error **en validación**, no en la prueba. Si el error no empeora, la variable sobraba. "
     "Dos ejemplos: en el TP2, sacar el mes y el trimestre casi no cambia el error en validación (98,8 contra 103,7 GWh), "
     "porque la estacionalidad ya entra por la demanda del año anterior; en el agrupamiento de agentes, «caídas frecuentes» repetía la volatilidad (correlación 0,77) y al sacarla "
     "la silueta sube de 0,30 a 0,44. Con pocos datos, además, Ridge regulariza: achica los coeficientes de las variables "
     "que no ayudan. La tarifa la sacamos por contexto, no por el número: su cambio era de registro, no de consumo.",
     "TP2 · Jugar con las variables y Agrupamiento"),
    ("¿Aplicaron normalización? ¿Cómo la validaron?",
     "Sí. Los modelos lineales, la SVM, k vecinos y el agrupamiento usan `StandardScaler` (media 0, desvío 1), y las demandas "
     "pasan antes por un logaritmo. El escalado vive dentro de un `Pipeline`: en cada bloque de validación se ajusta solo con "
     "los meses de entrenamiento de ese bloque, así no le pasa información del futuro al modelo. Los árboles no lo necesitan. "
     "En Ridge la normalización importa porque la penalización depende de la escala; se puede ver apagándola.",
     "TP2 · Jugar con las variables (interruptor «Normalizar»)"),
    ("¿Cómo se ven las métricas y las curvas? ¿Cuándo el modelo descarta?",
     "El modelo da una probabilidad de «sube». Si supera el **umbral**, dice «sube»; si no, lo **descarta** y dice «baja». "
     "Al mover el umbral cambian la matriz de confusión, la precisión, la exhaustividad y el F1, y el punto se mueve sobre las "
     "curvas ROC y precisión-exhaustividad. En la prueba, todos los modelos quedan debajo de la diagonal del azar (AUC < 0,5): "
     "ordenan al revés, y ningún umbral lo arregla.",
     "TP3 · Umbral y curvas"),
    ("¿Qué representa cada grupo?",
     "Con las regiones y meses (TP3), k = 3: **Gran Buenos Aires**, el **resto de las regiones de enero a junio** y el "
     "**resto de julio a diciembre**. Son tamaño y estación, no tendencia (acuerdo con «sube o baja»: −0,01). Con los agentes, "
     "sin «caídas frecuentes», k = 3: un bloque grande de 706 agentes parejos, 119 agentes chicos y en crecimiento, y 43 "
     "agentes muy variables y en caída.",
     "Agrupamiento · Grupos y buscador"),
    ("Si soy el cliente, ¿cómo encuentro mi resultado?",
     "En **Agrupamiento**, el buscador «¿En qué grupo está mi cliente?» permite escribir el nombre del agente (por ejemplo "
     "EDENOR, Acindar o una cooperativa) y muestra su grupo, qué significa y cómo se compara con el promedio. Para un pronóstico "
     "de consumo de una región, está **TP2 · Probar el pronóstico**.",
     "Agrupamiento · Grupos y buscador"),
    ("¿Por qué eligieron Ridge y la SVM si otros modelos dieron mejor en la prueba?",
     "Porque se eligieron **antes** de abrir la prueba, con la validación. Elegir después al que mejor salió sería quedarse "
     "con el que tuvo suerte. Igual mostramos todos los modelos como análisis.",
     "TP2 · Comparar modelos"),
    ("¿Por qué fallan todos los modelos?",
     "La demanda cambió de tendencia: cayó 12 meses seguidos (sep 2018 – ago 2019) y en la prueba volvió a crecer. Las "
     "causas no están en el historial: el cierre de la explotación a cielo abierto de Minera Alumbrera (49 % de la caída del "
     "Noroeste), la recesión, los aumentos de tarifas y un invierno templado en 2019. Para el TP4 hacen falta esas variables.",
     "TP2 · Comparar modelos"),
]
for preg, resp, donde in P:
    with st.expander(preg):
        st.markdown(resp)
        st.caption(f"📍 Dónde verlo: {donde}")
