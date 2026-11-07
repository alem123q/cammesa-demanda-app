# Pronóstico de demanda eléctrica regional · CAMMESA

Aplicación web del **Trabajo Práctico 4** de *Aprendizaje Automático y Grandes Datos*
(Ingeniería en Computación, Universidad Nacional de Rafaela, 2026).
**Grupo 6:** Francisco Alemandi · Eloy Kirov · Sebastián Palavecino.

🔗 **Aplicación desplegada:** _(completar con la URL de Streamlit Community Cloud)_

## Qué hace

Pronostica la **demanda eléctrica del mes siguiente** (en GWh) para cada una de las 9 regiones del
Mercado Eléctrico Mayorista argentino, a partir de dos datos que se conocen con un mes de anticipación:
la demanda del mes anterior y la del mismo mes del año anterior.

- **Pronóstico individual:** región, año y mes; las demandas se completan con la historia de CAMMESA
  y se pueden editar. Muestra el pronóstico, la regla simple «igual que hace un año», el rango de error
  esperable y, si el mes está en los datos, el valor real.
- **Validación de entradas:** valores positivos, rangos por región (detecta errores de unidades),
  saltos poco habituales y meses fuera del período con datos.
- **Contexto:** gráfico interactivo con la historia de la región y el pronóstico; explicación de cuánto
  aporta cada variable; pestaña con el desempeño en la prueba, por región.
- **Lote:** carga de un CSV con muchos pronósticos y descarga de los resultados (hay plantilla de ejemplo).
- **Registro:** las predicciones guardadas quedan registradas con fecha, para analizar el uso y la deriva.

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
app.py                     aplicación Streamlit
entrenar.py                reconstruye y serializa el modelo a partir de los datos
datos/serie_regional.csv   demanda mensual por región (9 regiones × 38 meses), ya limpia
modelo/ridge_regional.joblib   modelo serializado (Pipeline completo)
modelo/metadatos.json      métricas de la prueba, rangos por región y coeficientes
registro/                  registro de predicciones (se crea al usarla)
requirements.txt           dependencias con versión fija
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
