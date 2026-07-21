# Introducción al clustering

El clustering es un tipo de [Aprendizaje No Supervisado](https://wikipedia.org/wiki/Unsupervised_learning) que presupone que un conjunto de datos no está etiquetado o que sus entradas no están emparejadas con salidas predefinidas. Utiliza varios algoritmos para clasificar datos no etiquetados y proporcionar agrupaciones según los patrones que discierne en los datos.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Haz clic en la imagen de arriba para ver un video. Mientras estudias machine learning con clustering, disfruta de algunos temas de Nigerian Dance Hall — esta es una canción muy popular de 2014 de PSquare.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

### Introducción

El [Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) es muy útil para la exploración de datos. Veamos si puede ayudar a descubrir tendencias y patrones en la forma en que el público nigeriano consume música.

✅ Tómate un minuto para pensar en los usos del clustering. En la vida real, el clustering ocurre cada vez que tienes un montón de ropa sucia y necesitas separar las prendas de los miembros de tu familia 🧦👕👖🩲. En ciencia de datos, el clustering ocurre cuando se intenta analizar las preferencias de un usuario, o determinar las características de cualquier conjunto de datos no etiquetado. El clustering, en cierto modo, ayuda a darle sentido al caos, como un cajón de calcetines.

[![Introducción al ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introducción al Clustering")

> 🎥 Haz clic en la imagen de arriba para ver un video: John Guttag del MIT presenta el clustering

En un entorno profesional, el clustering se puede utilizar para determinar cosas como la segmentación de mercado, por ejemplo, determinar qué grupos de edad compran qué artículos. Otro uso sería la detección de anomalías, quizás para detectar fraudes en un conjunto de datos de transacciones con tarjetas de crédito. O podrías usar clustering para determinar tumores en un lote de exploraciones médicas.

✅ Piensa un minuto cómo podrías haber encontrado clustering 'en la naturaleza', en un entorno bancario, de comercio electrónico o empresarial.

> 🎓 Curiosamente, el análisis de clusters se originó en los campos de la Antropología y la Psicología en la década de 1930. ¿Te imaginas cómo podría haberse utilizado?

Alternativamente, podrías usarlo para agrupar resultados de búsqueda — por enlaces de compras, imágenes o reseñas, por ejemplo. El clustering es útil cuando tienes un conjunto de datos grande que deseas reducir y sobre el cual deseas realizar un análisis más granular, por lo que la técnica se puede utilizar para aprender sobre los datos antes de construir otros modelos.

✅ Una vez que tus datos están organizados en clusters, les asignas un ID de cluster, y esta técnica puede ser útil para preservar la privacidad de un conjunto de datos; puedes referirte a un punto de datos por su ID de cluster, en lugar de por datos identificables más reveladores. ¿Puedes pensar en otras razones por las que preferirías referirte a un ID de cluster en lugar de a otros elementos del cluster para identificarlo?

Profundiza tu comprensión de las técnicas de clustering en este [módulo de Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Comenzando con el clustering

[Scikit-learn ofrece una gran variedad](https://scikit-learn.org/stable/modules/clustering.html) de métodos para realizar clustering. El tipo que elijas dependerá de tu caso de uso. Según la documentación, cada método tiene varios beneficios. Aquí hay una tabla simplificada de los métodos compatibles con Scikit-learn y sus casos de uso apropiados:

| Nombre del método            | Caso de uso                                                           |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | propósito general, inductivo                                         |
| Affinity propagation         | muchos clusters desiguales, inductivo                                |
| Mean-shift                   | muchos clusters desiguales, inductivo                                |
| Spectral clustering          | pocos clusters uniformes, transductivo                               |
| Ward hierarchical clustering | muchos clusters restringidos, transductivo                           |
| Agglomerative clustering     | muchos, restringido, distancias no euclidianas, transductivo         |
| DBSCAN                       | geometría no plana, clusters desiguales, transductivo                |
| OPTICS                       | geometría no plana, clusters desiguales con densidad variable, transductivo |
| Gaussian mixtures            | geometría plana, inductivo                                           |
| BIRCH                        | conjunto de datos grande con valores atípicos, inductivo             |

> 🎓 Cómo creamos los clusters tiene mucho que ver con cómo agrupamos los puntos de datos. Desglosemos algo de vocabulario:
>
> 🎓 ['Transductivo' vs. 'inductivo'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> La inferencia transductiva se deriva de casos de entrenamiento observados que se asignan a casos de prueba específicos. La inferencia inductiva se deriva de casos de entrenamiento que se asignan a reglas generales que luego se aplican a casos de prueba.
> 
> Un ejemplo: imagina que tienes un conjunto de datos que solo está parcialmente etiquetado. Algunas cosas son 'discos', algunos 'cds' y otros están en blanco. Tu trabajo es proporcionar etiquetas para los espacios en blanco. Si eliges un enfoque inductivo, entrenarías un modelo buscando 'discos' y 'cds', y aplicarías esas etiquetas a tus datos no etiquetados. Este enfoque tendrá problemas para clasificar cosas que en realidad son 'cassettes'. Un enfoque transductivo, por otro lado, maneja estos datos desconocidos de manera más efectiva ya que trabaja para agrupar elementos similares y luego aplica una etiqueta a un grupo. En este caso, los clusters podrían reflejar 'cosas musicales redondas' y 'cosas musicales cuadradas'.
> 
> 🎓 ['Geometría no plana' vs. 'geometría plana'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivado de la terminología matemática, la geometría no plana vs. plana se refiere a la medida de distancias entre puntos mediante métodos geométricos 'planos' ([Euclideana](https://wikipedia.org/wiki/Euclidean_geometry)) o 'no planos' (no euclideana).
>
> 'Plano' en este contexto se refiere a la geometría euclideana (partes de la cual se enseñan como geometría 'plana'), y no plano se refiere a la geometría no euclideana. ¿Qué tiene que ver la geometría con el machine learning? Bueno, como dos campos que están arraigados en las matemáticas, debe haber una forma común de medir distancias entre puntos en los clusters, y eso se puede hacer de manera 'plana' o 'no plana', dependiendo de la naturaleza de los datos. Las [distancias euclideanas](https://wikipedia.org/wiki/Euclidean_distance) se miden como la longitud de un segmento de línea entre dos puntos. Las [distancias no euclideanas](https://wikipedia.org/wiki/Non-Euclidean_geometry) se miden a lo largo de una curva. Si tus datos, visualizados, parecen no existir en un plano, es posible que necesites usar un algoritmo especializado para manejarlos.
>
![Infografía de geometría plana vs no plana](./images/flat-nonflat.png)
> Infografía de [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distancias'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Los clusters se definen por su matriz de distancias, es decir, las distancias entre puntos. Esta distancia se puede medir de varias maneras. Los clusters euclideanos se definen por el promedio de los valores de los puntos y contienen un 'centroide' o punto central. Las distancias se miden, por lo tanto, por la distancia a ese centroide. Las distancias no euclideanas se refieren a 'clusteroides', el punto más cercano a otros puntos. Los clusteroides a su vez se pueden definir de varias maneras.
> 
> 🎓 ['Restringido'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> El [Clustering Restringido](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduce el aprendizaje 'semi-supervisado' en este método no supervisado. Las relaciones entre puntos se marcan como 'no puede enlazar' o 'debe enlazar', por lo que se imponen algunas reglas en el conjunto de datos.
>
> Un ejemplo: si un algoritmo se libera sobre un lote de datos no etiquetados o semi-etiquetados, los clusters que produce pueden ser de mala calidad. En el ejemplo anterior, los clusters podrían agrupar 'cosas musicales redondas' y 'cosas musicales cuadradas' y 'cosas triangulares' y 'galletas'. Si se dan algunas restricciones o reglas a seguir ("el elemento debe ser de plástico", "el elemento debe poder producir música"), esto puede ayudar a 'restringir' el algoritmo para tomar mejores decisiones.
> 
> 🎓 'Densidad'
> 
> Los datos que son 'ruidosos' se consideran 'densos'. Las distancias entre puntos en cada uno de sus clusters pueden resultar, al examinarlas, más o menos densas, o 'aglomeradas', y por lo tanto estos datos deben analizarse con el método de clustering apropiado. [Este artículo](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demuestra la diferencia entre usar los algoritmos K-Means y HDBSCAN para explorar un conjunto de datos ruidoso con densidad de cluster desigual.

## Algoritmos de clustering

Hay más de 100 algoritmos de clustering, y su uso depende de la naturaleza de los datos en cuestión. Analicemos algunos de los principales:

- **Clustering jerárquico**. Si un objeto se clasifica por su proximidad a un objeto cercano, en lugar de a uno más lejano, los clusters se forman en función de la distancia de sus miembros hacia y desde otros objetos. El clustering aglomerativo de Scikit-learn es jerárquico.

   ![Infografía de clustering jerárquico](./images/hierarchical.png)
   > Infografía de [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering de centroides**. Este popular algoritmo requiere la elección de 'k', o el número de clusters a formar, después de lo cual el algoritmo determina el punto central de un cluster y agrupa los datos alrededor de ese punto. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) es una versión popular del clustering de centroides. El centro se determina por la media más cercana, de ahí el nombre. La distancia al cuadrado desde el cluster se minimiza.

   ![Infografía de clustering de centroides](./images/centroid.png)
   > Infografía de [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering basado en distribución**. Basado en modelos estadísticos, el clustering basado en distribución se centra en determinar la probabilidad de que un punto de datos pertenezca a un cluster y asignarlo en consecuencia. Los métodos de mezcla gaussiana pertenecen a este tipo.

- **Clustering basado en densidad**. Los puntos de datos se asignan a clusters según su densidad o su agrupación entre sí. Los puntos de datos lejos del grupo se consideran valores atípicos o ruido. DBSCAN, Mean-shift y OPTICS pertenecen a este tipo de clustering.

- **Clustering basado en cuadrícula**. Para conjuntos de datos multidimensionales, se crea una cuadrícula y los datos se dividen entre las celdas de la cuadrícula, creando así clusters.

## Ejercicio - agrupa tus datos

El clustering como técnica se beneficia enormemente de una visualización adecuada, así que comencemos visualizando nuestros datos musicales. Este ejercicio nos ayudará a decidir cuál de los métodos de clustering debemos usar de manera más efectiva para la naturaleza de estos datos.

1. Abre el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) en esta carpeta.

1. Importa el paquete `Seaborn` para una buena visualización de datos.

    ```python
    !pip install seaborn
    ```

1. Agrega los datos de canciones de [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Carga un dataframe con algunos datos sobre las canciones. Prepárate para explorar estos datos importando las librerías y mostrando los datos:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Verifica las primeras líneas de datos:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtén información sobre el dataframe llamando a `info()`:

    ```python
    df.info()
    ```

   La salida se ve así:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Verifica la presencia de valores nulos llamando a `isnull()` y confirmando que la suma sea 0:

    ```python
    df.isnull().sum()
    ```

    Se ve bien:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Describe los datos:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Si estamos trabajando con clustering, un método no supervisado que no requiere datos etiquetados, ¿por qué mostramos estos datos con etiquetas? En la fase de exploración de datos, son útiles, pero no son necesarias para que los algoritmos de clustering funcionen. Podrías eliminar los encabezados de las columnas y referirte a los datos por el número de columna.

Observa los valores generales de los datos. Nota que la popularidad puede ser '0', lo que muestra canciones que no tienen clasificación. Eliminemos esas pronto.

1. Usa un gráfico de barras para descubrir los géneros más populares:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![más popular](./images/popular.png)

✅ Si deseas ver más valores principales, cambia el `[:5]` a un valor más grande, o elimínalo para ver todos.

Observa que cuando el género principal se describe como 'Missing', significa que Spotify no lo clasificó, así que eliminémoslo.

1. Elimina los datos faltantes filtrándolos:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Ahora vuelve a verificar los géneros:

    ![más popular](images/all-genres.png)

1. Con diferencia, los tres géneros principales dominan este conjunto de datos. Concentrémonos en `afro dancehall`, `afropop` y `nigerian pop`, y además filtremos el conjunto de datos para eliminar cualquier elemento con valor de popularidad 0 (lo que significa que no fue clasificado con una popularidad en el conjunto de datos y puede considerarse ruido para nuestros propósitos):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Haz una prueba rápida para ver si los datos se correlacionan de alguna manera particularmente fuerte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlaciones](images/correlation.png)

    La única correlación fuerte es entre `energy` y `loudness`, lo cual no es sorprendente, dado que la música fuerte suele ser bastante energética. Por lo demás, las correlaciones son relativamente débiles. Será interesante ver qué puede hacer un algoritmo de clustering con estos datos.

    > 🎓 Ten en cuenta que correlación no implica causalidad. Tenemos evidencia de correlación pero no evidencia de causalidad. Un [sitio web divertido](https://tylervigen.com/spurious-correlations) tiene algunas visualizaciones que enfatizan este punto.

¿Existe alguna convergencia en este conjunto de datos en torno a la popularidad percibida y la bailabilidad de una canción? Un FacetGrid muestra que hay círculos concéntricos que se alinean, independientemente del género. ¿Será posible que los gustos nigerianos converjan en un cierto nivel de bailabilidad para este género?

✅ Prueba diferentes puntos de datos (energy, loudness, speechiness) y más géneros musicales o diferentes. ¿Qué puedes descubrir? Echa un vistazo a la tabla `df.describe()` para ver la distribución general de los puntos de datos.

### Ejercicio - distribución de datos

¿Son estos tres géneros significativamente diferentes en la percepción de su bailabilidad, según su popularidad?

1. Examina la distribución de datos de nuestros tres géneros principales en cuanto a popularidad y bailabilidad a lo largo de un eje x e y determinados.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Puedes descubrir círculos concéntricos alrededor de un punto general de convergencia, que muestran la distribución de puntos.

    > 🎓 Ten en cuenta que este ejemplo utiliza un gráfico KDE (Estimación de Densidad Kernel) que representa los datos usando una curva de densidad de probabilidad continua. Esto nos permite interpretar datos cuando trabajamos con múltiples distribuciones.

    En general, los tres géneros se alinean de manera aproximada en términos de su popularidad y bailabilidad. Determinar clusters en estos datos con alineación tan laxa será un desafío:

    ![distribución](images/distribution.png)

1. Crea un diagrama de dispersión:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un diagrama de dispersión de los mismos ejes muestra un patrón similar de convergencia

    ![Facetgrid](images/facetgrid.png)

En general, para clustering, puedes usar diagramas de dispersión para mostrar clusters de datos, por lo que dominar este tipo de visualización es muy útil. En la próxima lección, tomaremos estos datos filtrados y usaremos clustering k-means para descubrir grupos en estos datos que parecen superponerse de maneras interesantes.

---

## 🚀Desafío

En preparación para la próxima lección, haz un gráfico sobre los diversos algoritmos de clustering que podrías descubrir y usar en un entorno de producción. ¿Qué tipo de problemas está tratando de abordar el clustering?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Antes de aplicar algoritmos de clustering, como hemos aprendido, es una buena idea entender la naturaleza de tu conjunto de datos. Lee más sobre este tema [aquí](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Este artículo útil](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) te guía a través de las diferentes formas en que varios algoritmos de clustering se comportan, dados diferentes formas de datos.

## Asignación

[Investiga otras visualizaciones para clustering](assignment.md)
