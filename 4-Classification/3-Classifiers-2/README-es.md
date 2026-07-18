# Clasificadores de cocinas 2

En esta segunda lección de clasificación, explorarás más formas de clasificar datos numéricos. También aprenderás sobre las implicaciones de elegir un clasificador sobre otro.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

### Prerrequisito

Asumimos que has completado las lecciones anteriores y que tienes un conjunto de datos limpio en tu carpeta `data` llamado _cleaned_cuisines.csv_ en la raíz de esta carpeta de 4 lecciones.

### Preparación

Hemos cargado tu archivo _notebook.ipynb_ con el conjunto de datos limpio y lo hemos dividido en dataframes X e y, listos para el proceso de construcción del modelo.

## Un mapa de clasificación

Anteriormente, aprendiste sobre las diversas opciones que tienes al clasificar datos usando la hoja de referencia de Microsoft. Scikit-learn ofrece una hoja de referencia similar, pero más detallada, que puede ayudarte aún más a reducir tus estimadores (otro término para clasificadores):

![Mapa de ML de Scikit-learn](images/map.png)
> Consejo: [visita este mapa en línea](https://scikit-learn.org/stable/tutorial/machine_learning_map/) y haz clic a lo largo de la ruta para leer la documentación.

### El plan

Este mapa es muy útil una vez que tienes una comprensión clara de tus datos, ya que puedes 'caminar' a lo largo de sus rutas hacia una decisión:

- Tenemos >50 muestras
- Queremos predecir una categoría
- Tenemos datos etiquetados
- Tenemos menos de 100K muestras
- ✨ Podemos elegir un Linear SVC
- Si eso no funciona, como tenemos datos numéricos
    - Podemos probar un ✨ KNeighbors Classifier
      - Si eso no funciona, prueba ✨ SVC y ✨ Ensemble Classifiers

Esta es una ruta muy útil a seguir.

## Ejercicio - dividir los datos

Siguiendo esta ruta, deberíamos empezar importando algunas librerías para usar.

1. Importa las librerías necesarias:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Divide tus datos de entrenamiento y prueba:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Clasificador Linear SVC

Support-Vector Clustering (SVC) es un descendiente de la familia de técnicas de ML de Máquinas de Vectores de Soporte (aprende más sobre ellas más abajo). En este método, puedes elegir un 'kernel' para decidir cómo agrupar las etiquetas. El parámetro 'C' se refiere a la 'regularización', que regula la influencia de los parámetros. El kernel puede ser uno de [varios](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); aquí lo configuramos como 'linear' para asegurar que aprovechamos Linear SVC. Probability tiene valor predeterminado 'false'; aquí lo configuramos como 'true' para obtener estimaciones de probabilidad. Establecemos el random state en '0' para mezclar los datos y obtener probabilidades.

### Ejercicio - aplicar un Linear SVC

Comienza creando un arreglo de clasificadores. Irás agregando progresivamente a este arreglo a medida que probamos.

1. Comienza con un Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Entrena tu modelo usando Linear SVC e imprime un informe:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    El resultado es bastante bueno:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Clasificador K-Neighbors

K-Neighbors es parte de la familia de métodos de ML "vecinos", que puede usarse tanto para aprendizaje supervisado como no supervisado. En este método, se crea un número predefinido de puntos y los datos se agrupan alrededor de estos puntos de manera que se puedan predecir etiquetas generalizadas para los datos.

### Ejercicio - aplicar el clasificador K-Neighbors

El clasificador anterior fue bueno y funcionó bien con los datos, pero quizás podemos obtener mejor precisión. Prueba un clasificador K-Neighbors.

1. Agrega una línea a tu arreglo de clasificadores (agrega una coma después del elemento Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    El resultado es un poco peor:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Aprende sobre [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Los clasificadores de vectores de soporte son parte de la familia de métodos de ML [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) que se utilizan para tareas de clasificación y regresión. Las SVM "mapean ejemplos de entrenamiento a puntos en el espacio" para maximizar la distancia entre dos categorías. Los datos subsiguientes se mapean en este espacio para que se pueda predecir su categoría.

### Ejercicio - aplicar un Support Vector Classifier

Intentemos obtener una precisión un poco mejor con un Support Vector Classifier.

1. Agrega una coma después del elemento K-Neighbors, y luego agrega esta línea:

    ```python
    'SVC': SVC(),
    ```

    ¡El resultado es bastante bueno!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Aprende sobre [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Sigamos la ruta hasta el final, aunque la prueba anterior fue bastante buena. Probemos algunos 'Ensemble Classifiers', específicamente Random Forest y AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

El resultado es muy bueno, especialmente para Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Aprende sobre [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Este método de Machine Learning "combina las predicciones de varios estimadores base" para mejorar la calidad del modelo. En nuestro ejemplo, usamos Random Trees y AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), un método de promediado, construye un 'bosque' de 'árboles de decisión' infusionados con aleatoriedad para evitar el sobreajuste. El parámetro n_estimators se establece en el número de árboles.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ajusta un clasificador a un conjunto de datos y luego ajusta copias de ese clasificador al mismo conjunto de datos. Se centra en los pesos de los elementos clasificados incorrectamente y ajusta el ajuste para el siguiente clasificador para corregirlos.

---

## 🚀Desafío

Cada una de estas técnicas tiene una gran cantidad de parámetros que puedes ajustar. Investiga los parámetros predeterminados de cada una y reflexiona sobre qué significaría ajustar estos parámetros para la calidad del modelo.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Hay mucha jerga técnica en estas lecciones, así que tómate un minuto para revisar [esta lista](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) de terminología útil.

## Asignación

[Ajuste de parámetros](assignment.md)
