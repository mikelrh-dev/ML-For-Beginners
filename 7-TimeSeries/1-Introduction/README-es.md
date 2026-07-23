# Introducción a la predicción de series temporales

![Resumen de series temporales en sketchnote](../../sketchnotes/ml-timeseries.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

En esta lección y la siguiente, aprenderás sobre **predicción de series temporales**, una parte interesante y valiosa del repertorio de un científico de ML que es un poco menos conocida que otros temas. La predicción de series temporales es una especie de **"bola de cristal"**: basándote en el rendimiento pasado de una variable como el precio, puedes predecir su valor futuro potencial.

[![Introducción a la predicción de series temporales](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introducción a la predicción de series temporales")

> 🎥 Haz clic en la imagen de arriba para ver un video sobre predicción de series temporales

## [Quiz previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

Es un campo útil e interesante con valor real para los negocios, dada su aplicación directa a problemas de precios, inventario y cadena de suministro. Aunque las técnicas de deep learning han empezado a usarse para obtener más insights y predecir mejor el rendimiento futuro, la predicción de series temporales sigue siendo un campo muy influenciado por técnicas clásicas de ML.

> El plan de estudios de series temporales de Penn State está disponible [aquí](https://online.stat.psu.edu/stat510/lesson/1)

## Introducción

Imagina que mantienes un array de parquímetros inteligentes que proporcionan datos sobre con qué frecuencia se usan y durante cuánto tiempo a lo largo del tiempo.

> ¿Qué pasaría si pudieras predecir, basándote en el rendimiento pasado del parquímetro, su valor futuro según las leyes de oferta y demanda?

Predecir con precisión cuándo actuar para lograr tu objetivo es un desafío que se puede abordar con predicción de series temporales. No haría feliz a la gente que le cobren más en horas punta cuando buscan aparcamiento, ¡pero sería una forma segura de generar ingresos para limpiar las calles!

Exploremos algunos tipos de algoritmos de series temporales y empecemos un notebook para limpiar y preparar datos. Los datos que analizarás provienen de la **competición de pronóstico GEFCom2014**. Consisten en 3 años de valores horarios de carga eléctrica y temperatura entre 2012 y 2014. Dados los patrones históricos de carga eléctrica y temperatura, puedes predecir valores futuros de carga eléctrica.

En este ejemplo, aprenderás a pronosticar **un paso temporal adelante**, usando solo datos históricos de carga. Antes de empezar, es útil entender qué ocurre entre bastidores.

## Algunas definiciones

Cuando te encuentres con el término "serie temporal", necesitas entender su uso en varios contextos diferentes.

🎓 **Serie temporal**

En matemáticas, "una serie temporal es una serie de puntos de datos indexados (o listados o graficados) en orden temporal. Más comúnmente, una serie temporal es una secuencia tomada en puntos sucesivos igualmente espaciados en el tiempo". Un ejemplo de serie temporal es el valor de cierre diario del [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). El uso de gráficos de series temporales y modelado estadístico se encuentra frecuentemente en procesamiento de señales, predicción meteorológica, predicción de terremotos y otros campos donde ocurren eventos y los puntos de datos pueden graficarse a lo largo del tiempo.

🎓 **Análisis de series temporales**

El análisis de series temporales es el análisis de los datos de series temporales mencionados arriba. Los datos de series temporales pueden tomar formas distintas, incluyendo "series temporales interrumpidas" que detectan patrones en la evolución de una serie temporal antes y después de un evento interruptor. El tipo de análisis necesario depende de la naturaleza de los datos. Los datos de series temporales en sí pueden tomar la forma de series de números o caracteres.

El análisis a realizar usa una variedad de métodos, incluyendo dominio de frecuencia y dominio temporal, lineal y no lineal, y más. [Más información](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) sobre las muchas formas de analizar este tipo de datos.

🎓 **Predicción de series temporales**

La predicción de series temporales es el uso de un modelo para predecir valores futuros basándose en patrones mostrados por datos recopilados previamente en el pasado. Aunque es posible usar modelos de regresión para explorar datos de series temporales, con índices temporales como variables x en un gráfico, tales datos se analizan mejor usando tipos especiales de modelos.

Los datos de series temporales son una lista de observaciones ordenadas, a diferencia de los datos que se pueden analizar por regresión lineal. El más común es **ARIMA**, acrónimo de "Autoregressive Integrated Moving Average" (Promedio Móvil Integrado Autorregresivo).

Los [modelos ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relacionan el valor presente de una serie con valores pasados y errores de predicción pasados". Son más apropiados para analizar datos de dominio temporal, donde los datos están ordenados en el tiempo.

> Hay varios tipos de modelos ARIMA, que puedes aprender [aquí](https://people.duke.edu/~rnau/411arim.htm) y tocarás en la siguiente lección.

En la siguiente lección, construirás un modelo ARIMA usando **Series Temporales Univariadas**, que se centran en una variable que cambia su valor en el tiempo. Un ejemplo de este tipo de datos es [este dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) que registra la concentración mensual de CO2 en el Observatorio Mauna Loa:

|  CO2   | AñoMes | Año  | Mes |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identifica la variable que cambia en el tiempo en este dataset

## Características de los datos de series temporales a considerar

Al mirar datos de series temporales, podrías notar que tienen [ciertas características](https://online.stat.psu.edu/stat510/lesson/1/1.1) que necesitas tener en cuenta y mitigar para entender mejor sus patrones. Si consideras los datos de series temporales como potencialmente proporcionando una "señal" que quieres analizar, estas características pueden pensarse como "ruido". A menudo necesitarás reducir este "ruido" compensando algunas de estas características usando técnicas estadísticas.

Aquí hay conceptos que debes conocer para trabajar con series temporales:

🎓 **Tendencias**

Las tendencias se definen como aumentos y disminuciones medibles a lo largo del tiempo. [Más información](https://machinelearningmastery.com/time-series-trends-in-python). En el contexto de series temporales, se trata de cómo usar y, si es necesario, eliminar tendencias de tu serie temporal.

🎓 **[Estacionalidad](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

La estacionalidad se define como fluctuaciones periódicas, como las prisas navideñas que pueden afectar las ventas, por ejemplo. [Mira](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) cómo diferentes tipos de gráficos muestran estacionalidad en los datos.

🎓 **Outliers (valores atípicos)**

Los outliers están lejos de la varianza estándar de los datos.

🎓 **Ciclo a largo plazo**

Independiente de la estacionalidad, los datos pueden mostrar un ciclo a largo plazo, como una recesión económica que dura más de un año.

🎓 **Varianza constante**

Con el tiempo, algunos datos muestran fluctuaciones constantes, como el uso de energía por día y noche.

🎓 **Cambios abruptos**

Los datos pueden mostrar un cambio abrupto que podría necesitar análisis adicional. El cierre abrupto de negocios debido al COVID, por ejemplo, causó cambios en los datos.

✅ Aquí tienes un [gráfico de serie temporal de ejemplo](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) mostrando moneda del juego gastada diariamente durante unos años. ¿Puedes identificar alguna de las características listadas arriba en estos datos?

![Gasto de moneda en juego](./images/currency.png)

## Ejercicio - empezando con datos de consumo eléctrico

Empecemos a crear un modelo de series temporales para predecir el consumo eléctrico futuro dado el consumo pasado.

> Los datos en este ejemplo provienen de la competición de pronóstico GEFCom2014. Consisten en 3 años de valores horarios de carga eléctrica y temperatura entre 2012 y 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. En la carpeta `working` de esta lección, abre el archivo _notebook.ipynb_. Empieza añadiendo librerías que te ayudarán a cargar y visualizar datos:

   ```python
   import os
   import matplotlib.pyplot as plt
   from common.utils import load_data
   %matplotlib inline
   ```

   Nota: estás usando los archivos de la carpeta `common` incluida que configuran tu entorno y manejan la descarga de datos.

2. A continuación, examina los datos como un dataframe llamando a `load_data()` y `head()`:

   ```python
   data_dir = './data'
   energy = load_data(data_dir)[['load']]
   energy.head()
   ```

   Puedes ver que hay dos columnas representando fecha y carga:

   |                     |  load  |
   | :-----------------: | :----: |
   | 2012-01-01 00:00:00 | 2698.0 |
   | 2012-01-01 01:00:00 | 2558.0 |
   | 2012-01-01 02:00:00 | 2444.0 |
   | 2012-01-01 03:00:00 | 2402.0 |
   | 2012-01-01 04:00:00 | 2403.0 |

3. Ahora, grafica los datos llamando a `plot()`:

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![gráfico de energía](images/energy-plot.png)

4. Ahora, grafica la primera semana de julio 2014, proporcionándola como entrada a `energy` en el patrón `[fecha inicial]: [fecha final]`:

   ```python
   energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![julio](images/july-2014.png)

   ¡Un gráfico bonito! Mira estos gráfados y ve si puedes determinar alguna de las características listadas arriba. ¿Qué podemos deducir visualizando los datos?

En la siguiente lección, crearás un modelo ARIMA para hacer algunos pronósticos.

---

## 🚀 Desafío

Haz una lista de todas las industrias y áreas de investigación que se te ocurran que se beneficiarían de la predicción de series temporales. ¿Puedes pensar en una aplicación de estas técnicas en las artes? ¿En econometría? ¿Ecología? ¿Comercio minorista? ¿Industria? ¿Finanzas? ¿Dónde más?

## [Quiz posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Repaso y autoestudio

Aunque no los cubriremos aquí, las redes neuronales a veces se usan para mejorar métodos clásicos de predicción de series temporales. Lee más sobre ellas [en este artículo](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Tarea

[Visualiza más series temporales](assignment.md)