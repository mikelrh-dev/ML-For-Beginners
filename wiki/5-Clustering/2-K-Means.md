# Lección 2: K-Means Clustering

## ¿Qué aprendemos aquí?

K-Means es el algoritmo de clustering más popular. En esta lección lo aplicamos a nuestro dataset de música nigeriana y evaluamos si funciona bien (spoiler: no mucho, y eso es una lección importante).

## ¿Cómo funciona K-Means?

### Los 5 pasos
```
1. Elegís k (número de clusters)
2. El algoritmo coloca k centroids aleatoriamente
3. Asigna cada punto al centroid más cercano
4. Recalcula los centroids como el promedio de sus puntos
5. Repite los pasos 3-4 hasta que se estabilicen
```

### Analogía visual
Imaginá que tenés 3 pileta de pelotas de colores 🎱🎱🎱:
- **Paso 1**: Elegís 3 pileta (k=3)
- **Paso 2**: Colocás 3 centroids (centros) aleatoriamente
- **Paso 3**: Cada pelota va a la pileta más cercana
- **Paso 4**: Recalculás el centro de cada pileta
- **Paso 5**: Repetís hasta que las pelotas no cambien de pileta

### Visualización
Cada cluster se puede ver como una región en un **diagrama de Voronoi** — cada región es un cluster, y cada centroid es el "centro" de su región.

## Paso 1: Preparar los datos

### ¿Por qué necesitamos LabelEncoder?
K-Means **no entiende texto** — solo números. Entonces convertimos los géneros:
```
afro dancehall  → 0
afropop         → 1
nigerian pop    → 2
```

### ¿Qué columnas usamos?
Seleccionamos 6 características para comparar canciones:
- `artist_top_genre` (codificado como número)
- `popularity`
- `danceability`
- `acousticness`
- `loudness`
- `energy`

### Ejemplo de código
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X = df.loc[:, ('artist_top_genre','popularity','danceability',
               'acousticness','loudness','energy')]
X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
```

## Paso 2: Silhouette Score — ¿Qué tan buenos son los clusters?

### ¿Qué es?
El Silhouette Score mide qué tan bien separados están los clusters. Va de **-1 a 1**.

### ¿Cómo se interpreta?

| Score | Significado | Analogía |
|-------|-------------|----------|
| **1** | Perfecto — clusters bien separados | 3 mesas en esquinas opuestas de un salón |
| **0.7 - 1** | Muy bien — clusters claros | 3 grupos en diferentes áreas |
| **0.5 - 0.7** | Regular — hay superposición | Gente sentada entre mesas |
| **0 - 0.5** | Malo — clusters difusos | Todos mezclados en el centro |
| **-1** | Pésimo — puntos en cluster equivocado | Alguien en la mesa equivocada |

### La pregunta clave
Para cada punto, el score pregunta: "¿Estás más cerca de la gente de TU cluster o del cluster más cercano?"

### Ejemplo de código
```python
from sklearn.cluster import KMeans
from sklearn import metrics

km = KMeans(n_clusters=3, random_state=0)
km.fit(X)
y_cluster_kmeans = km.predict(X)

score = metrics.silhouette_score(X, y_cluster_kmeans)
print(f"Silhouette Score: {score:.3f}")
```

### Nuestro resultado: ~0.53
Está en la **zona media-baja**. Los clusters existen, pero no están bien definidos. Es como si en una fiesta, la gente de diferentes grupos estuviera medio mezclada.

## Paso 3: Método del codo — ¿Cuántos clusters usar?

### El problema
¿Cómo sabés si usar 3, 5 o 10 clusters? El método del codo te da una **evidencia visual**.

### ¿Cómo funciona?
1. Probás k de 1 a 10
2. Graficás la **inercia** (WCSS) — qué tan compactos son los clusters
3. Buscás el **"codo"** — donde deja de bajar rápido

### ¿Qué es WCSS?
**Within-Cluster Sum of Squares** — mide la suma de distancias al cuadrado de cada punto a su centroide.

- **Bajo** = clusters apretaditos (bueno)
- **Alto** = clusters dispersos (malo)

### Ejemplo de código
```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()
```

### ¿Cómo leer el gráfico?

```
Inercia
  │
  │\ 
  │ \ 
  │  \____
  │       \____
  └──────────────→ k
     1  2  3  4  5
         ↑
      El codo (k=3)
```

**El codo** indica el k óptimo — agregar más clusters ya no mejora mucho.

### Analogía
Es como preguntar "¿cuántas mesas necesito en el restaurante?" Si tenés 30 comensales:
- 1 mesa → todos apretados (malo)
- 3 mesas → cómodos (bien)
- 10 mesas → sobran mesas (innecesario)

El codo te dice cuándo parar.

### ¿Qué es `k-means++`?
Una forma inteligente de colocar los centroids iniciales — los coloca lejos entre sí, lo que generalmente da mejores resultados que aleatorio.

## Paso 4: Visualizar los clusters

### Ejemplo de código
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(df['popularity'], df['danceability'], c=labels)
plt.xlabel('Popularity')
plt.ylabel('Danceability')
plt.show()
```

### ¿Qué vemos?
Los 3 colores (clusters) están **superpuestos** — no hay separación clara. K-Means fuerza 3 grupos, pero no son naturales.

## Paso 5: Evaluar la precisión

### Ejemplo de código
```python
labels = kmeans.labels_
correct_labels = sum(y == labels)
print(f"Resultado: {correct_labels} de {y.size} muestras correctas")
print(f"Precisión: {correct_labels/float(y.size):.2f}")
```

### ¿Qué nos dice?
La precisión es baja. Comparamos los clusters predichos con los géneros reales y el algoritmo no acierta mucho.

## ¿Por qué la precisión es baja?

| Problema | Explicación |
|----------|-------------|
| **Poca correlación** | Las características no están muy relacionadas entre sí |
| **Outliers** | Valores extremos distorsionan los centroids |
| **Superposición** | Los géneros musicales se mezclan en estas dimensiones |
| **Varianza alta** | Los datos están demasiado dispersos |

### Lección importante
> **No todos los datasets son buenos candidatos para K-Means.** Si los datos no tienen estructura de cluster natural, el algoritmo va a forzar agrupaciones artificiales.

## ¿Qué podrías mejorar?

| Mejora | Cómo ayuda |
|--------|------------|
| **Escalar los datos** (StandardScaler) | Todas las columnas tienen el mismo peso |
| **Usar otras columnas** | Quizás instrumentalness o speechiness ayuden |
| **Probar otro algoritmo** | DBSCAN o Gaussian Mixture pueden funcionar mejor |
| **Limpiar más outliers** | Reducir el ruido en los datos |

## Errores comunes

1. **No escalar los datos**: Si una columna tiene rango 0-1 y otra 0-1000, la segunda domina
2. **No probar diferentes k**: Siempre usá el método del codo
3. **Confiar solo en la precisión**: El silhouette score también importa
4. **No visualizar**: Siempre mirá los clusters antes de concluir

## Analogía final

Es como intentar separar colores en un cuadro que tiene mucho blur 🎨. Si los colores están muy mezclados, no importa cuántas veces reorganices — no vas a lograr separación nítida.

## Resumen de la sección

1. **Visualización** (lección 1): Exploramos y limpiamos los datos
2. **K-Means** (esta lección): Aplicamos el algoritmo y evaluamos
3. **Resultado**: Los datos no son buenos candidatos para clustering con estas características

**La lección más importante:** La calidad de los datos importa más que la elección del algoritmo.

---

**Siguiente sección:** [6-NLP](../../6-NLP/README.md) — Procesamiento de lenguaje natural
