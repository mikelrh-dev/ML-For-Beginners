# Regression with Scikit-learn

## Instructions

Take a look at the [Linnerud dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud) in Scikit-learn. This dataset has multiple [targets](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset): 'It consists of three exercise (data) and three physiological (target) variables collected from twenty middle-aged men in a fitness club'.

In your own words, describe how to create a Regression model that would plot the relationship between the waistline and how many situps are accomplished. Do the same for the other datapoints in this dataset.

## Rubric

| Criteria                       | Exemplary                           | Adequate                      | Needs Improvement          |
| ------------------------------ | ----------------------------------- | ----------------------------- | -------------------------- |
| Submit a descriptive paragraph | Well-written paragraph is submitted | A few sentences are submitted | No description is supplied |


Para crear un modelo de regresión con el dataset de Linnerud que relacione la línea de la cintura con la cantidad de abdominales, el proceso seguiría la misma lógica estructurada de Scikit-learn. Primero, importaríamos el dataset usando load_linnerud(). A diferencia de otros conjuntos de datos, Linnerud contiene matrices multivariantes tanto para las características como para los objetivos.

Para nuestro modelo, aislaríamos la variable independiente ($X$) extrayendo la segunda columna (índice 1) de la matriz de datos fisiológicos (linnerud.target), que corresponde a la cintura (waistline), y le aplicaríamos un .reshape((-1, 1)) para convertir el vector plano en una matriz bidimensional compatible con Scikit-learn. El objetivo a predecir ($y$) sería la segunda columna (índice 1) de la matriz de ejercicios (linnerud.data), que registra los abdominales (situps).


A continuación, dividiríamos los datos de los 20 sujetos en conjuntos de entrenamiento y prueba mediante train_test_split(). Instanciaríamos un objeto de LinearRegression() y entrenaríamos el modelo con el método .fit(X_train, y_train) para que la IA calcule la línea recta de mejor ajuste matemático entre ambas variables. Finalmente, usaríamos .predict(X_test) para evaluar el rendimiento y generaríamos una gráfica de dispersión con Matplotlib cruzando los puntos reales con la línea de predicción azul. Para el resto de puntos de datos del dataset (como relacionar el peso con las dominadas, o el pulso con los saltos), se replicaría exactamente el mismo flujo de ingeniería, modificando únicamente los índices de las columnas al recortar las matrices originales para analizar las nuevas correlaciones.