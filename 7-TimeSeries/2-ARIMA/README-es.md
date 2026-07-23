# Pronóstico de series temporales con ARIMA

En la lección anterior, aprendiste los fundamentos del pronóstico de series temporales y cargaste un dataset que muestra las fluctuaciones de la carga eléctrica a lo largo del tiempo.

[![Introducción a ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introducción a ARIMA")

> 🎥 Haz clic en la imagen para ver un video: Breve introducción a modelos ARIMA. El ejemplo está en R, pero los conceptos son universales.

## [Quiz previo a la lección](https://ff-quizzes.netlify.app/es/ml/)

## Introducción

En esta lección descubrirás una forma específica de construir modelos con **ARIMA: *A*uto*R*egresivo *I*ntegrado *M*edia *M*óvil** (AutoRegressive Integrated Moving Average). Los modelos ARIMA son especialmente adecuados para ajustar datos que muestran **no-estacionariedad**.

## Conceptos generales

Para trabajar con ARIMA, necesitas conocer estos conceptos:

- 🎓 **Estacionariedad**. En estadística, la estacionariedad se refiere a datos cuya distribución no cambia al desplazarse en el tiempo. Los datos no-estacionarios muestran fluctuaciones debidas a tendencias que deben transformarse para analizarlos. La estacionalidad, por ejemplo, puede introducir fluctuaciones y eliminarse mediante "diferenciación estacional".

- 🎓 **[Diferenciación](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. La diferenciación es el proceso de transformar datos no-estacionarios para hacerlos estacionarios, eliminando su tendencia no constante. "La diferenciación elimina los cambios en el nivel de una serie temporal, eliminando tendencia y estacionalidad y estabilizando consecuentemente la media de la serie temporal." [Paper de Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA en el contexto de series temporales

Desempaquetemos los componentes de ARIMA para entender cómo ayuda a modelar series temporales y hacer predicciones.

- **AR - AutoRegresivo**. Los modelos autoregresivos, como su nombre indica, miran "hacia atrás" en el tiempo para analizar valores previos y hacer suposiciones sobre ellos. Estos valores previos se llaman "rezagos" (lags). Un ejemplo serían datos de ventas mensuales de lápices. El total de ventas de cada mes sería una "variable en evolución" en el dataset. Este modelo se construye como "la variable de interés en evolución es regresada sobre sus propios valores rezagados (previos)." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrado**. A diferencia de los modelos "ARMA" similares, la "I" en ARIMA se refiere a su aspecto *integrado*. Los datos están "integrados" cuando se aplican pasos de diferenciación para eliminar la no-estacionariedad.

- **MA - Media Móvil**. El aspecto de [media móvil](https://wikipedia.org/wiki/Moving-average_model) se refiere a la variable de salida determinada observando los valores actuales y pasados de los rezagos.

**En resumen**: ARIMA se usa para ajustar un modelo lo más cerca posible a la forma especial de los datos de series temporales.

## Ejercicio - Construir un modelo ARIMA

Abre la carpeta `_/working_` en esta lección y encuentra el archivo `_notebook.ipynb_`.

1. Ejecuta el notebook para cargar la librería Python `statsmodels`; la necesitarás para modelos ARIMA.

2. Carga las librerías necesarias

3. Carga más librerías útiles para graficar datos:

```python
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from common.utils import load_data, mape
from IPython.display import Image

%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore") # ignorar mensajes de advertencia
```

4. Carga los datos desde `/data/energy.csv` en un DataFrame de Pandas y examínalo:

```python
energy = load_data('./data')[['load']]
energy.head(10)
```

5. Grafica todos los datos de energía disponibles desde enero 2012 a diciembre 2014. No debería haber sorpresas ya que vimos estos datos en la lección anterior:

```python
energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()
```

Ahora, ¡construyamos un modelo!

### Crear datasets de entrenamiento y prueba

Ahora tienes los datos cargados, así que puedes separarlos en conjuntos de entrenamiento y prueba. Entrenarás tu modelo con el conjunto de entrenamiento. Como de costumbre, después de que el modelo termine de entrenar, evaluarás su precisión usando el conjunto de prueba. Debes asegurarte de que el conjunto de prueba cubra un período posterior en el tiempo al del conjunto de entrenamiento para garantizar que el modelo no gane información de períodos futuros.

1. Asigna un período de dos meses del 1 de septiembre al 31 de octubre de 2014 al conjunto de entrenamiento. El conjunto de prueba incluirá el período de dos meses del 1 de noviembre al 31 de diciembre de 2014:

```python
train_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'
```

Como estos datos reflejan el consumo diario de energía, hay un patrón estacional fuerte, pero el consumo es más similar al de días recientes.

1. Visualiza las diferencias:

```python
energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
    .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()
```

![datos de entrenamiento y prueba](images/train-test.png)

Por tanto, usar una ventana de tiempo relativamente pequeña para entrenar los datos debería ser suficiente.

> Nota: Como la función que usamos para ajustar el modelo ARIMA usa validación in-sample durante el ajuste, omitiremos los datos de validación.

### Preparar los datos para el entrenamiento

Ahora necesitas preparar los datos para el entrenamiento filtrando y escalando. Filtra tu dataset para incluir solo los períodos y columnas necesarios, y escala para asegurar que los datos estén proyectados en el intervalo (0, 1).

1. Filtra el dataset original para incluir solo los períodos mencionados por conjunto y solo la columna necesaria 'load' más la fecha:

```python
train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
test = energy.copy()[energy.index >= test_start_dt][['load']]

print('Forma de datos de entrenamiento: ', train.shape)
print('Forma de datos de prueba: ', test.shape)
```

Puedes ver la forma de los datos:

```output
Forma de datos de entrenamiento:  (1416, 1)
Forma de datos de prueba:  (48, 1)
```

2. Escala los datos al rango (0, 1).

```python
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
train.head(10)
```

3. Visualiza los datos originales vs. escalados:

```python
energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'carga original'}).plot.hist(bins=100, fontsize=12)
train.rename(columns={'load':'carga escalada'}).plot.hist(bins=100, fontsize=12)
plt.show()
```

![original](images/original.png)
> Los datos originales

![escalados](images/scaled.png)
> Los datos escalados

4. Ahora que has calibrado los datos escalados, puedes escalar los datos de prueba:

```python
test['load'] = scaler.transform(test)
test.head()
```

### Implementar ARIMA

¡Es hora de implementar ARIMA! Ahora usarás la librería `statsmodels` que instalaste antes.

Debes seguir varios pasos:

1. Define el modelo llamando a `SARIMAX()` y pasando los parámetros: p, d, q, y P, D, Q.
2. Prepara el modelo para los datos de entrenamiento llamando a `fit()`.
3. Haz predicciones llamando a `forecast()` y especificando el número de pasos (el `horizonte`) a pronosticar.

> 🎓 **¿Para qué sirven todos estos parámetros?** En un modelo ARIMA hay 3 parámetros que modelan los aspectos principales de una serie temporal: estacionalidad, tendencia y ruido. Estos parámetros son:

`p`: parámetro asociado al aspecto auto-regresivo, que incorpora valores *pasados*.
`d`: parámetro asociado a la parte integrada, que afecta la cantidad de *diferenciación* (¿recuerdas la diferenciación 👆?) a aplicar.
`q`: parámetro asociado a la parte de media móvil del modelo.

> Nota: Si tus datos tienen un aspecto estacional (como este caso), usamos un modelo ARIMA estacional (SARIMA). En ese caso necesitas otro conjunto de parámetros: `P`, `D`, `Q` que describen las mismas asociaciones que `p`, `d`, `q`, pero corresponden a los componentes estacionales del modelo.

1. Empieza definiendo tu valor de horizonte preferido. Probemos 3 horas:

```python
# Especificar el número de pasos a pronosticar hacia adelante
HORIZON = 3
print('Horizonte de pronóstico:', HORIZON, 'horas')
```

Seleccionar los mejores valores para los parámetros de un modelo ARIMA puede ser desafiante ya que es algo subjetivo y requiere tiempo. Podrías considerar usar una función `auto_arima()` de la librería [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Por ahora prueba selecciones manuales para encontrar un buen modelo.

```python
order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)

model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
results = model.fit()

print(results.summary())
```

Se imprime una tabla de resultados.

¡Has construido tu primer modelo! Ahora necesitamos evaluarlo.

### Evaluar tu modelo

Para evaluar el modelo, puedes realizar la validación **walk-forward** (hacia adelante). En la práctica, los modelos de series temporales se re-entrenan cada vez que un nuevo dato está disponible. Esto permite al modelo hacer el mejor pronóstico en cada paso temporal.

Partiendo del inicio de la serie temporal con esta técnica, entrenas el modelo con el dataset de entrenamiento. Luego haces una predicción en el siguiente paso temporal. La predicción se evalúa contra el valor conocido. El conjunto de entrenamiento se expande para incluir el valor conocido y el proceso se repite.

> Nota: Deberías mantener la ventana de entrenamiento fija para un entrenamiento más eficiente, de modo que cada vez que añadas una nueva observación al conjunto de entrenamiento, elimines la observación del inicio del conjunto.

Este proceso proporciona una estimación más robusta de cómo funcionará el modelo en la práctica. Sin embargo, tiene el costo computacional de crear tantos modelos. Es aceptable si los datos son pequeños o el modelo es simple, pero podría ser un problema a escala.

La validación walk-forward es el estándar de oro para evaluar modelos de series temporales y se recomienda para tus propios proyectos.

1. Primero, crea un punto de datos de prueba para cada paso del HORIZONTE.

```python
test_shifted = test.copy()

for t in range(1, HORIZON+1):
    test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

test_shifted = test_shifted.dropna(how='any')
test_shifted.head(5)
```

|            |          | load | load+1 | load+2 |
| ---------- | -------- | ---- | ------ | ------ |
| 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
| 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
| 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
| 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
| 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

Los datos se desplazan horizontalmente según su punto de horizonte.

2. Haz predicciones en tus datos de prueba usando este enfoque de ventana deslizante en un bucle del tamaño de la longitud de los datos de prueba:

```python
%%time
training_window = 720 # dedicar 30 días (720 horas) para entrenamiento

train_ts = train['load']
test_ts = test_shifted

history = [x for x in train_ts]
history = history[(-training_window):]

predictions = list()

order = (2, 1, 0)
seasonal_order = (1, 1, 0, 24)

for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps = HORIZON)
    predictions.append(yhat)
    obs = list(test_ts.iloc[t])
    # mover la ventana de entrenamiento
    history.append(obs[0])
    history.pop(0)
    print(test_ts.index[t])
    print(t+1, ': predicho =', yhat, 'esperado =', obs)
```

Puedes ver el entrenamiento ocurriendo:

```output
2014-12-30 00:00:00
1 : predicho = [0.32 0.29 0.28] esperado = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

2014-12-30 01:00:00
2 : predicho = [0.3  0.29 0.3 ] esperado = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

2014-12-30 02:00:00
3 : predicho = [0.27 0.28 0.32] esperado = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
```

3. Compara las predicciones con la carga real:

```python
eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediccion', var_name='h')
eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
eval_df[['prediccion', 'actual']] = scaler.inverse_transform(eval_df[['prediccion', 'actual']])
eval_df.head()
```

Salida
|     |            | timestamp | h   | prediccion | actual   |
| --- | ---------- | --------- | --- | ---------- | -------- |
| 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
| 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
| 2   | 2014-12-30 | 02:00:00  | t+2 | 2,900.17   | 2,899.00 |
| 3   | 2014-12-30 | 03:00:00  | t+3 | 2,917.69   | 2,886.00 |
| 4   | 2014-12-30 | 04:00:00  | t+4 | 2,946.99   | 2,963.00 |

Observa la predicción de los datos horarios comparada con la carga real. ¿Qué tan precisa es?

### Verificar la precisión del modelo

Verifica la precisión de tu modelo probando su error porcentual absoluto medio (MAPE) sobre todas las predicciones.

> **🧮 Muéstrame las matemáticas**
>
> ![MAPE](images/mape.png)
>
>  El [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) muestra la precisión de la predicción como una razón definida por la fórmula anterior. La diferencia entre actual<sub>t</sub> y predicho<sub>t</sub> se divide por actual<sub>t</sub>. "El valor absoluto en este cálculo se suma para cada punto pronosticado en el tiempo y se divide por el número de puntos ajustados n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)

1. Expresa la ecuación en código:

```python
if(HORIZON > 1):
    eval_df['APE'] = (eval_df['prediccion'] - eval_df['actual']).abs() / eval_df['actual']
    print(eval_df.groupby('h')['APE'].mean())
```

2. Calcula el MAPE de un paso:

```python
print('MAPE pronóstico a un paso: ', (mape(eval_df[eval_df['h'] == 't+1']['prediccion'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
```

    MAPE pronóstico a un paso:  0.5570581332313952 %

3. Imprime el MAPE del pronóstico multi-paso:

```python
print('MAPE pronóstico multi-paso: ', mape(eval_df['prediccion'], eval_df['actual'])*100, '%')
```

```output
MAPE pronóstico multi-paso:  1.1460048657704118 %
```

Un número bajo es mejor: considera que un pronóstico con MAPE de 10 está equivocado en 10%.

4. Como siempre, es más fácil ver esta métrica visualmente, así que grafiquémosla:

```python
 if(HORIZON == 1):
     ## Graficar pronóstico a un paso
     eval_df.plot(x='timestamp', y=['actual', 'prediccion'], style=['r', 'b'], figsize=(15, 8))

 else:
     ## Graficar pronóstico multi-paso
     plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
     for t in range(1, HORIZON+1):
         plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediccion'].values

     fig = plt.figure(figsize=(15, 8))
     ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
     ax = fig.add_subplot(111)
     for t in range(1, HORIZON+1):
         x = plot_df['timestamp'][(t-1):]
         y = plot_df['t+'+str(t)][0:len(x)]
         ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

     ax.legend(loc='best')

 plt.xlabel('timestamp', fontsize=12)
 plt.ylabel('load', fontsize=12)
 plt.show()
 ```

![un modelo de serie temporal](images/accuracy.png)

🏆 ¡Un gráfico muy bueno, mostrando un modelo con buena precisión. ¡Bien hecho!

---

## 🚀 Desafío

Investiga las formas de probar la precisión de un modelo de series temporales. Tocamos MAPE en esta lección, ¿hay otros métodos? Investígalos y anótalos. Un documento útil se encuentra [aquí](https://otexts.com/fpp2/accuracy.html)

## [Quiz posterior a la lección](https://ff-quizzes.netlify.app/es/ml/)

## Repaso y autoestudio

Esta lección toca solo los fundamentos del Pronóstico de Series Temporales con ARIMA. Tómate tiempo para profundizar explorando [este repositorio](https://microsoft.github.io/forecasting/) y sus varios tipos de modelos para aprender otras formas de construir modelos de series temporales.

## Tarea

[Un nuevo modelo ARIMA](assignment.md)