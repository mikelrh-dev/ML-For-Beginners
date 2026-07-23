# Predicción de series temporales con Support Vector Regressor

En la lección anterior aprendiste a usar el modelo ARIMA para hacer predicciones de series temporales. Ahora verás el modelo **Support Vector Regressor**, un modelo de regresión usado para predecir datos continuos.

## [Quiz previo a la lección](https://ff-quizzes.netlify.app/es/ml/)

## Introducción

En esta lección, descubrirás una forma específica de construir modelos con [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) para regresión, o **SVR: Support Vector Regressor** (Regresor de Vectores Soporte).

### SVR en el contexto de series temporales [^1]

Antes de entender la importancia de SVR en predicción de series temporales, aquí tienes algunos conceptos importantes:

- **Regresión:** Técnica de aprendizaje supervisado para predecir valores continuos a partir de un conjunto de entradas. La idea es ajustar una curva (o línea) en el espacio de características que tenga la máxima cantidad de puntos de datos. [Más información](https://en.wikipedia.org/wiki/Regression_analysis).
- **Support Vector Machine (SVM):** Tipo de modelo de aprendizaje supervisado usado para clasificación, regresión y detección de outliers. El modelo es un hiperplano en el espacio de características, que en clasificación actúa como frontera y en regresión actúa como la línea de mejor ajuste. En SVM, se usa una función Kernel para transformar el dataset a un espacio de mayor dimensión, de modo que sean fácilmente separables. [Más sobre SVMs](https://en.wikipedia.org/wiki/Support-vector_machine).
- **Support Vector Regressor (SVR):** Un tipo de SVM para encontrar la línea de mejor ajuste (que en SVM es un hiperplano) con la máxima cantidad de puntos de datos.

### ¿Por qué SVR? [^1]

En la lección anterior aprendiste sobre ARIMA, un método estadístico lineal muy exitoso para pronosticar datos de series temporales. Sin embargo, en muchos casos, los datos de series temporales tienen **no-linealidad**, que no puede ser capturada por modelos lineales. En esos casos, la capacidad de SVM para considerar la no-linealidad en los datos para tareas de regresión hace que SVR sea exitoso en predicción de series temporales.

## Ejercicio - Construir un modelo SVR

Los primeros pasos de preparación de datos son los mismos que en la lección anterior sobre [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA).

Abre la carpeta [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) de esta lección y encuentra el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb).[^2]

1. Ejecuta el notebook e importa las librerías necesarias: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. Carga los datos desde `/data/energy.csv` en un DataFrame de Pandas y echa un vistazo: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Grafica todos los datos disponibles de energía desde enero 2012 a diciembre 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![datos completos](images/full-data.png)

   Ahora, construyamos nuestro modelo SVR.

### Crear datasets de entrenamiento y prueba

Ahora tus datos están cargados, así que puedes separarlos en conjuntos de entrenamiento y prueba. Luego remodelarás los datos para crear un dataset basado en pasos temporales que necesitarás para SVR. Entrenarás tu modelo con el set de entrenamiento. Después de que el modelo termine el entrenamiento, evaluarás su precisión en el set de entrenamiento, set de prueba y luego el dataset completo para ver el rendimiento general. Necesitas asegurar que el set de prueba cubra un período posterior al de entrenamiento para que el modelo no gane información de períodos futuros [^2] (situación conocida como *Overfitting*).

1. Asigna un período de dos meses del 1 de septiembre al 31 de octubre 2014 al set de entrenamiento. El set de prueba incluirá el período de dos meses del 1 de noviembre al 31 de diciembre 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualiza las diferencias: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![datos de entrenamiento y prueba](images/train-test.png)

### Preparar los datos para entrenamiento

Ahora necesitas preparar los datos para entrenamiento realizando filtrado y escalado. Filtra tu dataset para incluir solo los períodos y columnas que necesitas, y escala para asegurar que los datos estén en el intervalo 0,1.

1. Filtra el dataset original para incluir solo los períodos mencionados por set y solo la columna necesaria 'load' más la fecha: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Forma datos entrenamiento: ', train.shape)
   print('Forma datos prueba: ', test.shape)
   ```

   ```output
   Forma datos entrenamiento:  (1416, 1)
   Forma datos prueba:  (48, 1)
   ```
    
2. Escala los datos de entrenamiento al rango (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
    
3. Ahora, escala los datos de prueba: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Crear datos con pasos temporales [^1]

Para SVR, transformas los datos de entrada para que tengan la forma `[batch, timesteps]`. Así que remodelas los `train_data` y `test_data` existentes para que haya una nueva dimensión que se refiera a los timesteps.

```python
# Convertir a arrays numpy
train_data = train.values
test_data = test.values
```

Para este ejemplo, usamos `timesteps = 5`. Así que las entradas al modelo son los datos de los primeros 4 timesteps, y la salida será el dato del 5º timestep.

```python
timesteps=5
```

Convertir datos de entrenamiento a tensor 2D usando comprensión de listas anidada:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Convertir datos de prueba a tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Seleccionar entradas y salidas de datos de entrenamiento y prueba:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### Implementar SVR [^1]

Ahora, implementemos SVR. Para leer más sobre esta implementación, consulta [esta documentación](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Para nuestra implementación, seguimos estos pasos:

1. Definir el modelo llamando `SVR()` y pasando los hiperparámetros: kernel, gamma, C y epsilon
2. Preparar el modelo para los datos de entrenamiento llamando `fit()`
3. Hacer predicciones llamando `predict()`

Ahora creamos un modelo SVR. Usamos el [kernel RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), y establecemos los hiperparámetros gamma, C y epsilon como 0.5, 10 y 0.05 respectivamente.

```python
model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
```

#### Entrenar el modelo con datos de entrenamiento [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Hacer predicciones del modelo [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

¡Has construido tu SVR! Ahora necesitamos evaluarlo.

### Evaluar tu modelo [^1]

Para la evaluación, primero escalamos de vuelta los datos a su escala original. Luego, para verificar el rendimiento, graficaremos la serie temporal original y predicha, e imprimiremos el resultado MAPE.

Escalar las predicciones y valores originales:

```python
# Escalar las predicciones
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Escalar los valores originales
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Verificar rendimiento en datos de entrenamiento y prueba [^1]

Extraemos los timestamps del dataset para mostrar en el eje x del gráfico. Nota que usamos los primeros `timesteps-1` valores como entrada para la primera salida, así que los timestamps para la salida empezarán después de eso.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Graficar predicciones para datos de entrenamiento:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color='red', linewidth=2.0, alpha=0.6)
plt.plot(train_timestamps, y_train_pred, color='blue', linewidth=0.8)
plt.legend(['Actual','Predicho'])
plt.xlabel('Timestamp')
plt.title("Predicción datos de entrenamiento")
plt.show()
```

![predicción datos entrenamiento](images/train-data-predict.png)

Imprimir MAPE para datos de entrenamiento

```python
print('MAPE datos entrenamiento: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE datos entrenamiento: 1.7195710200875551 %
```

Graficar predicciones para datos de prueba

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color='red', linewidth=2.0, alpha=0.6)
plt.plot(test_timestamps, y_test_pred, color='blue', linewidth=0.8)
plt.legend(['Actual','Predicho'])
plt.xlabel('Timestamp')
plt.show()
```

![predicción datos prueba](images/test-data-predict.png)

Imprimir MAPE para datos de prueba

```python
print('MAPE datos prueba: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE datos prueba:  1.2623790187854018 %
```

🏆 ¡Tienes un resultado muy bueno en el dataset de prueba!

### Verificar rendimiento del modelo en dataset completo [^1]

```python
# Extraer valores de carga como array numpy
data = energy.copy().values

# Escalar
data = scaler.transform(data)

# Transformar a tensor 2D según requerimiento de entrada del modelo
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Forma tensor: ", data_timesteps.shape)

# Seleccionar entradas y salidas de los datos
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("Forma X: ", X.shape,"\nForma Y: ", Y.shape)
```

```output
Forma tensor:  (26300, 5)
Forma X:  (26300, 4) 
Forma Y:  (26300, 1)
```

```python
# Hacer predicciones del modelo
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale y remodelar
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color='red', linewidth=2.0, alpha=0.6)
plt.plot(Y_pred, color='blue', linewidth=0.8)
plt.legend(['Actual','Predicho'])
plt.xlabel('Timestamp')
plt.show()
```

![predicción datos completos](images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 ¡Muy buenos gráficos, mostrando un modelo con buena precisión. ¡Bien hecho!

---

## 🚀 Desafío

- Intenta ajustar los hiperparámetros (gamma, C, epsilon) al crear el modelo y evalúa en los datos para ver qué conjunto da mejores resultados en el set de prueba. Para saber más sobre estos hiperparámetros, consulta [este documento](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel).
- Intenta usar diferentes funciones kernel para el modelo y analiza su rendimiento en el dataset. Un documento útil [aquí](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Prueba diferentes valores de `timesteps` para que el modelo mire más atrás y haga la predicción.

## [Quiz posterior a la lección](https://ff-quizzes.netlify.app/es/ml/)

## Repaso y autoestudio

Esta lección introdujo la aplicación de SVR para Predicción de Series Temporales. Para leer más sobre SVR, consulta [este blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Esta [documentación de scikit-learn](https://scikit-learn.org/stable/modules/svm.html) proporciona una explicación más completa sobre SVMs en general, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) y también otros detalles de implementación como las diferentes [funciones kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) que se pueden usar y sus parámetros.

## Tarea

[Un nuevo modelo SVR](assignment.md)

## Créditos

[^1]: El texto, código y salida en esta sección fue contribuido por [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: El texto, código y salida en esta sección se tomó de [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)