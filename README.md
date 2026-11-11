# Pronóstico de demanda eléctrica regional · CAMMESA

Aplicación web del **Trabajo Práctico 4** de *Aprendizaje Automático y Grandes Datos*
(Ingeniería en Computación, Universidad Nacional de Rafaela, 2026).
**Grupo 6:** Francisco Alemandi · Eloy Kirov · Sebastián Palavecino.

🔗 **Aplicación desplegada:** _(completar con la URL de Streamlit Community Cloud)_

## Qué hace

Es un tablero con cuatro secciones, pensado para que alguien sin conocimientos de programación pueda probar los
modelos y entender los resultados:

| Sección | Página | Qué se puede hacer |
|---|---|---|
| Inicio | Inicio | Qué resuelve el proyecto, los resultados en tres números, cómo leer cada término y qué variables se usan. |
| **TP2 · Regresión** | Probar el pronóstico | Elegir región y mes y ver el pronóstico de Ridge junto a la regla «igual que hace un año»; pronóstico en lote desde un CSV; registro de predicciones. |
| | Comparar modelos | Tabla de los 6 modelos y 3 reglas; validación contra prueba; pronósticos de cada modelo contra lo real, por región. |
| | Jugar con las variables | Sacar o agregar variables, prender o apagar la normalización y mover α: Ridge se reentrena al instante. |
| **TP3 · Clasificación** | Probar «sube o baja» | Para una región y un mes, qué dijo cada modelo, qué dijo la persistencia y qué pasó. |
| | Umbral y curvas | Mover el umbral y ver la matriz de confusión, las métricas y el punto sobre las curvas ROC y precisión-exhaustividad. |
| | Comparar modelos | Tabla de los 8 clasificadores, F1 en validación y prueba, sobreajuste y matrices de confusión. |
| Agrupamiento | Grupos y buscador | Curva de la silueta, qué representa cada grupo y buscador «¿en qué grupo está mi región / mi cliente?», por región y mes o por agente (868 series). |
| Ayuda | Preguntas frecuentes | Respuestas cortas a las preguntas de la cátedra. |

Validación de entradas: valores positivos, rangos por región (detecta errores de unidades), saltos poco
habituales y meses fuera del período con datos.

## El modelo

| | |
|---|---|
| Algoritmo | Regresión **Ridge** (α = 3,16), elegida y fijada antes de abrir la prueba (TP2) |
| Predictoras | región (indicadores), demanda del mes anterior, demanda del mismo mes del año anterior, mes, trimestre |
| Preprocesamiento | log(1 + x) de las demandas y del objetivo, escalado; todo dentro de un `Pipeline` |
| Entrenamiento | enero 2018 – agosto 2019 (180 meses-región) |
| Prueba | septiembre 2019 – febrero 2020 (54 meses-región), abierta una sola vez |
| RMSE en la prueba | **76,7 GWh** (regla «mismo mes del año anterior»: **71,0 GWh**) |

**Lectura honesta:** el modelo no superó a la regla simple (la meta era 63,9 GWh). Aprendió en un período de
caída de la demanda y se evaluó cuando volvía a crecer. Por eso la aplicación muestra siempre los dos valores.

## Estructura

```
app.py                         punto de entrada: navegación entre páginas
comun.py                       carga de datos, formato y colores compartidos
paginas/                       una página por sección del tablero
entrenar.py                    reconstruye y serializa el modelo Ridge (modelo/ridge_regional.joblib)
preparar_resultados.py         reproduce TP2, TP3 y el agrupamiento y guarda los resultados en resultados/
datos/serie_regional.csv       demanda mensual por región (9 regiones × 38 meses), ya limpia
datos/agentes_perfiles.csv     perfil y grupo de cada serie agente-tarifa (análisis por agente)
modelo/                        modelo serializado y metadatos (métricas, rangos, coeficientes)
resultados/                    métricas, pronósticos y probabilidades de validación y prueba
requirements.txt               dependencias con versión fija
```

## Ejecutar localmente

Requiere Python 3.10 a 3.12.

```bash
git clone https://github.com/alem123q/cammesa-demanda-app.git
cd cammesa-demanda-app
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python entrenar.py                 # opcional: vuelve a generar el modelo (da RMSE 76,7 / 71,0 GWh)
python preparar_resultados.py      # opcional: vuelve a calcular los resultados del TP2 y el TP3 (unos 20 s)
streamlit run app.py
```

## Datos

CAMMESA, «Demanda últimos años», Portal de datos abiertos de energía, Secretaría de Energía de la Nación
(datos.energia.gob.ar). Se excluyó la categoría «Mercado a Término» (761 de sus 766 filas son valores de
relleno de −0,001 a −0,003 MWh) y se sumó la demanda de todos los agentes por región y mes.

## Limitaciones

- Solo usa la historia de la demanda: no conoce temperatura, actividad económica ni tarifas.
- Los datos terminan en febrero de 2020; pronosticar más allá es un ejercicio, no un pronóstico confiable.
- En Streamlit Community Cloud el disco no es persistente: el registro de predicciones se pierde al
  reiniciarse el servidor (se puede descargar desde la pestaña «Registro»).
