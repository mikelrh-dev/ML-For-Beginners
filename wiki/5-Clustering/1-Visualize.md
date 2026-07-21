# Lección 1: Visualización de datos de música nigeriana

## ¿Qué aprendemos aquí?

Antes de aplicar clustering, necesitamos **entender nuestros datos**. Esta lección es como mirar el terreno antes de construir una casa — si no conocés la tierra, vas a tener problemas después.

**El objetivo:** Explorar, limpiar y visualizar un dataset de 530 canciones nigerianas de Spotify para entender si es buen candidato para clustering.

## El dataset

Tenemos 530 canciones nigerianas de Spotify con 16 columnas:
- **Texto**: nombre, álbum, artista, género
- **Números**: popularidad, bailabilidad, acústica, energía, ruido, tempo, etc.

## Paso 1: Exploración básica

### ¿Qué hacemos y por qué?

| Comando | Qué hace | Por qué lo usamos |
|---------|----------|-------------------|
| `df.head()` | Muestra las primeras 5 filas | Ver rápido qué tiene el dataset |
| `df.info()` | Muestra tipos de datos y nulos | Verificar que no hay datos faltantes |
| `df.isnull().sum()` | Cuenta nulos por columna | Confirmar que el dataset está limpio |
| `df.describe()` | Estadísticas resumidas | Ver distribución y detectar outliers |

### Ejemplo de código
```python
df = pd.read_csv("../data/nigerian-songs.csv")
df.head()          # Ver primeras filas
df.info()          # Ver estructura
df.isnull().sum()  # Verificar nulos
df.describe()      # Ver estadísticas
```

### Qué buscar
- **En `info()`**: ¿Hay columnas con menos datos que otras? (datos faltantes)
- **En `describe()`**: ¿El max es muy distinto del 75%? (outliers)
- **En `isnull()`**: ¿Algún valor mayor a 0? (datos faltantes)

## Paso 2: Limpieza de datos

### ¿Qué limpiamos y por qué?

| Acción | Por qué |
|--------|---------|
| Eliminar géneros 'Missing' | Spotify no los clasificó, son ruido |
| Filtrar 3 géneros principales | Nos concentramos en los más representativos |
| Eliminar popularidad 0 | Son datos sin clasificar, ruido para el clustering |

### Ejemplo de código
```python
# Eliminar 'Missing'
df = df[df['artist_top_genre'] != 'Missing']

# Filtrar 3 géneros principales
df = df[(df['artist_top_genre'] == 'afro dancehall') | 
        (df['artist_top_genre'] == 'afropop') | 
        (df['artist_top_genre'] == 'nigerian pop')]

# Eliminar popularidad 0
df = df[(df['popularity'] > 0)]
```

### Analogía
Es como quando ordenás una habitación 🧹 — primero sacás la basura (Missing), después organizás lo que queda (filtrar géneros), y finalmente quitás lo que no sirve (popularidad 0).

## Paso 3: Visualización

### 3.1 Gráfico de barras — Géneros más populares
```python
top = df['artist_top_genre'].value_counts()
sns.barplot(x=top.index, y=top.values)
```
**¿Qué hace?** Cuenta cuántas hay de cada género y las dibuja como barras.

### 3.2 Heatmap de correlación
```python
corrmat = df.corr(numeric_only=True)
sns.heatmap(corrmat, vmax=.8, square=True)
```
**¿Qué hace?** Muestra qué tan relacionadas están las columnas entre sí.

**¿Qué busca?**
- **Rojo fuerte** = alta correlación (las variables se mueven juntas)
- **Azul** = poca correlación (las variables son independientes)

**Nuestro resultado:** Solo `energy` y `loudness` tienen correlación fuerte (la música fuerte suele ser energética).

### 3.3 KDE (Kernel Density Estimate)
```python
sns.jointplot(data=df, x="popularity", y="danceability", 
              hue="artist_top_genre", kind="kde")
```
**¿Qué hace?** Muestra la **densidad** de puntos en 2 dimensiones con curvas de probabilidad.

**¿Qué busca?**
- **Círculos apretados** = muchos puntos concentrados ahí
- **Círculos abiertos** = pocos puntos
- **Colores mezclados** = los géneros se superponen

### 3.4 Scatter plot
```python
sns.FacetGrid(df, hue="artist_top_genre", height=5)
   .map(plt.scatter, "popularity", "danceability")
   .add_legend()
```
**¿Qué hace?** Dibuja cada canción como un punto, coloreado por género.

**¿Qué busca?** Si los colores están separados (bueno) o mezclados (malo para clustering).

## Resultado clave

Los tres géneros se **superponen bastante** en popularidad y bailabilidad. Esto significa que K-Means va a tener dificultades para separarlos — ¡eso es lo que exploramos en la lección 2!

**¿Por qué importa?** Si los datos no están bien separados naturalmente, ningún algoritmo va a mágicamente encontrar clusters perfectos.

## Errores comunes

1. **No limpiar datos faltantes**: Los 'Missing' arruinan los resultados
2. **No verificar outliers**: Los valores extremos distorsionan los centroids
3. **Confiar solo en números**: Siempre visualizá antes de clustering
4. **Usar todas las columnas**: Algunas no aportan información útil

## Analogía final

Es como mirar un mapa de temperaturas antes de decidir dónde poner aire acondicionado 🌡️. Si todos los colores están mezclados, no vas a poder crear zonas claras de temperatura.

---

**Siguiente:** [K-Means](2-K-Means.md) — Ahora sí aplicamos el algoritmo
