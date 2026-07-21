# Agrupamiento K-Means

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

En esta lección, aprenderás a crear agrupamientos usando Scikit-learn y el conjunto de datos de música nigeriana que importaste anteriormente. Cubriremos los fundamentos de K-Means para agrupamiento. Ten en cuenta que, como aprendiste en la lección anterior, hay muchas formas de trabajar con agrupamientos y el método que uses depende de tus datos. Probaremos K-Means por ser la técnica de agrupamiento más común. ¡Comencemos!

Términos que aprenderás:

- Puntaje de silueta
- Método del codo
- Inercia
- Varianza

## Introducción

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) es un método derivado del dominio del procesamiento de señales. Se utiliza para dividir y particionar grupos de datos en 'k' agrupamientos usando una serie de observaciones. Cada observación agrupa un punto de datos dado al 'mean' (media) más cercano, o el punto central de un agrupamiento.

Los agrupamientos se pueden visualizar como [diagramas de Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), que incluyen un punto (o 'semilla') y su región correspondiente.

![diagrama de voronoi](images/voronoi.png)

> infografía de [Jen Looper](https://twitter.com/jenlooper)

El proceso de agrupamiento K-Means [se ejecuta en un proceso de tres pasos](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. El algoritmo selecciona k cantidad de puntos centrales muestreando del conjunto de datos. Después de esto, itera:
    1. Asigna cada muestra al centroide más cercano.
    2. Crea nuevos centroides tomando el valor medio de todas las muestras asignadas a los centroides anteriores.
    3. Luego, calcula la diferencia entre los centroides nuevos y antiguos y repite hasta que los centroides se estabilizan.

Una desventaja de usar K-Means incluye el hecho de que necesitarás establecer 'k', es decir, el número de centroides. Afortunadamente, el 'método del codo' ayuda a estimar un buen valor inicial para 'k'. Lo probarás en un minuto.

## Prerrequisito

Trabajarás en el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) de esta lección, que incluye la importación de datos y la limpieza preliminar que realizaste en la lección anterior.

## Ejercicio - preparación

Comienza echando otro vistazo a los datos de las canciones.

1. Crea un boxplot (diagrama de caja) llamando a `boxplot()` para cada columna:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    Estos datos son un poco ruidosos: al observar cada columna como un boxplot, puedes ver valores atípicos (outliers).

    ![valores atípicos](images/boxplots.png)

Podrías revisar el conjunto de datos y eliminar estos valores atípicos, pero eso dejaría los datos bastante reducidos.

1. Por ahora, elige qué columnas usarás para tu ejercicio de agrupamiento. Escoge aquellas con rangos similares y codifica la columna `artist_top_genre` como datos numéricos:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Ahora necesitas elegir cuántos agrupamientos seleccionar. Sabes que hay 3 géneros musicales que extrajimos del conjunto de datos, así que probemos con 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Verás un arreglo impreso con los agrupamientos predichos (0, 1 o 2) para cada fila del dataframe.

1. Usa este arreglo para calcular un 'puntaje de silueta' (silhouette score):

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Puntaje de silueta

Busca un puntaje de silueta cercano a 1. Este puntaje varía de -1 a 1, y si el puntaje es 1, el agrupamiento es denso y está bien separado de otros agrupamientos. Un valor cercano a 0 representa agrupamientos superpuestos con muestras muy cercanas al límite de decisión de los agrupamientos vecinos. [(Fuente)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Nuestro puntaje es **.53**, justo en el medio. Esto indica que nuestros datos no son particularmente adecuados para este tipo de agrupamiento, pero continuemos.

### Ejercicio - construir un modelo

1. Importa `KMeans` e inicia el proceso de agrupamiento.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Hay algunas partes aquí que merecen explicación.

    > 🎓 range: Estas son las iteraciones del proceso de agrupamiento

    > 🎓 random_state: "Determina la generación de números aleatorios para la inicialización de centroides." [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" (suma de cuadrados dentro del agrupamiento) mide el promedio de la distancia al cuadrado de todos los puntos dentro de un agrupamiento al centroide del agrupamiento. [Fuente](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Inercia: Los algoritmos K-Means intentan elegir centroides para minimizar la 'inercia', "una medida de cuán internamente coherentes son los agrupamientos." [Fuente](https://scikit-learn.org/stable/modules/clustering.html). El valor se añade a la variable wcss en cada iteración.

    > 🎓 k-means++: En [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) puedes usar la optimización 'k-means++', que "inicializa los centroides para que estén (generalmente) distantes entre sí, lo que probablemente produce mejores resultados que la inicialización aleatoria."

### Método del codo

Anteriormente, supusiste que, debido a que seleccionaste 3 géneros musicales, deberías elegir 3 agrupamientos. ¿Pero es ese el caso?

1. Usa el 'método del codo' para asegurarte.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Usa la variable `wcss` que construiste en el paso anterior para crear un gráfico que muestre dónde está el 'codo', lo que indica el número óptimo de agrupamientos. ¡Quizás **sí** son 3!

    ![método del codo](images/elbow.png)

## Ejercicio - visualizar los agrupamientos

1. Intenta el proceso nuevamente, esta vez configurando tres agrupamientos, y muéstralos como un gráfico de dispersión:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. Verifica la precisión del modelo:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    La precisión de este modelo no es muy buena, y la forma de los agrupamientos te da una pista del porqué.

    ![agrupamientos](images/clusters.png)

    Estos datos están demasiado desbalanceados, tienen poca correlación y hay demasiada varianza entre los valores de las columnas para agrupar bien. De hecho, los agrupamientos que se forman probablemente están muy influenciados o sesgados por las tres categorías de género que definimos anteriormente. ¡Eso fue un proceso de aprendizaje!

    En la documentación de Scikit-learn, puedes ver que un modelo como este, con agrupamientos no muy bien demarcados, tiene un problema de 'varianza':

    ![modelos problemáticos](images/problems.png)
    > Infografía de Scikit-learn

## Varianza

La varianza se define como "el promedio de las diferencias al cuadrado respecto a la media" [(Fuente)](https://www.mathsisfun.com/data/standard-deviation.html). En el contexto de este problema de agrupamiento, se refiere a datos cuyos números tienden a divergir demasiado de la media.

✅ Este es un gran momento para pensar en todas las formas en que podrías corregir este problema. ¿Ajustar los datos un poco más? ¿Usar columnas diferentes? ¿Usar un algoritmo diferente? Pista: Intenta [escalar tus datos](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) para normalizarlos y probar otras columnas.

> Prueba esta '[calculadora de varianza](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' para entender el concepto un poco más.

---

## 🚀Desafío

Dedica tiempo a este notebook, ajustando parámetros. ¿Puedes mejorar la precisión del modelo limpiando más los datos (por ejemplo, eliminando valores atípicos)? Puedes usar pesos para dar más importancia a ciertas muestras de datos. ¿Qué más puedes hacer para crear mejores agrupamientos?

Pista: Intenta escalar tus datos. Hay código comentado en el notebook que agrega escalado estándar para que las columnas de datos se asemejen más entre sí en términos de rango. Encontrarás que, aunque el puntaje de silueta disminuye, el 'codo' en el gráfico se suaviza. Esto se debe a que dejar los datos sin escalar permite que los datos con menos varianza tengan más peso. Lee un poco más sobre este problema [aquí](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y estudio autónomo

Echa un vistazo a un simulador de K-Means [como este](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Puedes usar esta herramienta para visualizar puntos de datos de muestra y determinar sus centroides. Puedes editar la aleatoriedad de los datos, la cantidad de agrupamientos y la cantidad de centroides. ¿Te ayuda esto a entender cómo se pueden agrupar los datos?

Además, revisa [este documento sobre K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) de Stanford.

## Asignación

[Prueba diferentes métodos de agrupamiento](assignment.md)
