# Glosario de Clustering

## Términos clave

### Aprendizaje no supervisado
Tipo de machine learning que encuentra patrones en datos **sin etiquetas**. A diferencia del aprendizaje supervisado (donce tenés las respuestas correctas), el no supervisado debe descubrir la estructura por sí solo.

### Centroid
Punto central de un cluster en K-Means. Se calcula como el **promedio** de todos los puntos del cluster. Es como el "centro de masa" de un grupo.

### Cluster
Grupo de puntos de datos que son **similares entre sí** y diferentes a los de otros clusters. Es como una mesa en un restaurante — la gente de la misma mesa tiene cosas en común.

### Correlación
Relación entre dos variables. Va de -1 a 1:
- **1**: relación positiva perfecta (si una sube, la otra sube)
- **0**: sin relación
- **-1**: relación negativa perfecta (si una sube, la otra baja)

### Dendrograma
Diagrama de árbol que muestra la jerarquía de clusters en clustering jerárquico. Ayuda a visualizar cómo se agrupan los datos en diferentes niveles.

### Diagrama de Voronoi
Visualización que divide el espacio en regiones según la distancia a puntos centrales. Cada región representa un cluster.

### Distancia euclidiana
Medida de "distancia en línea recta" entre dos puntos. Es la más común en clustering: √[(x₂-x₁)² + (y₂-y₁)²]

### DBSCAN
Algoritmo de clustering basado en densidad. No necesita elegir k manualmente y puede detectar clusters de forma irregular y outliers.

### Elbow Method (Método del codo)
Técnica para elegir el número óptimo de clusters. Se grafica la inercia contra el número de clusters y se busca el "codo" donde la curva cambia de pendiente.

### Gaussian Mixture
Algoritmo de clustering que asume que los datos provienen de una mezcla de distribuciones normales. Es más flexible que K-Means pero más complejo.

### Heatmap (Mapa de calor)
Visualización que muestra valores numéricos como colores. En correlación, los colores cálidos (rojo) indican alta correlación y los fríos (azul) baja correlación.

### Inductive vs Transductive
- **Inductive** (inductivo): entrena reglas generales y las aplica a nuevos datos
- **Transductive** (transductivo): mapea casos observados a casos específicos, sin generar reglas generales

### Inertia (Inercia)
Medida de qué tan coherentes son los clusters internamente. Es la suma de distancias al cuadrado de cada punto a su centroide. Menor = mejor, pero siempre decrece con más clusters.

### K-Means
Algoritmo de clustering que divide los datos en k clusters asignando cada punto al centroid más cercano. Es el más popular pero requiere elegir k manualmente.

### K-Means++
Variante de K-Means que inicializa los centroids de forma inteligente (lejos entre sí), generalmente dando mejores resultados que la inicialización aleatoria.

### KDE (Kernel Density Estimate)
Gráfico que estima la función de densidad de probabilidad de una variable. Muestra dónde se concentran los datos como curvas suaves.

### LabelEncoder
Herramienta de scikit-learn que convierte etiquetas de texto a números. Por ejemplo: "gato"→0, "perro"→1, "pájaro"→2.

### Outlier (Valor atípico)
Dato que se aleja significativamente del resto. Puede distorsionar los resultados de clustering porque afecta los promedios.

### Silhouette Score
Mide qué tan bien separados están los clusters. Va de -1 a 1:
- **1**: clusters bien separados
- **0**: clusters superpuestos
- **-1**: puntos en cluster equivocado

### StandardScaler
Herramienta que estandariza las características para que tengan media 0 y desviación estándar 1. Importante para que todas las columnas tengan el mismo peso.

### WCSS (Within-Cluster Sum of Squares)
Suma de distancias al cuadrado de cada punto a su centroide. Mide qué tan compactos son los clusters. Se usa en el método del codo.

## Fórmulas importantes

### Distancia euclidiana
```
d = √[(x₂-x₁)² + (y₂-y₁)²]
```

### Silhouette Score para un punto
```
s = (b - a) / max(a, b)
```
Donde:
- `a` = distancia promedio a puntos del mismo cluster
- `b` = distancia promedio al cluster más cercano

### Inercia (WCSS)
```
WCSS = Σ||xᵢ - μₖ||²
```
Donde:
- `xᵢ` = cada punto del cluster
- `μₖ` = centroid del cluster k

## Paquetes de Python

| Paquete | Para qué sirve |
|---------|----------------|
| `pandas` | Manipulación de datos (DataFrames) |
| `matplotlib` | Gráficos básicos |
| `seaborn` | Gráficos estadísticos avanzados |
| `scikit-learn` | Algoritmos de machine learning |
| `LabelEncoder` | Convertir texto a números |
| `KMeans` | Algoritmo de clustering |
| `metrics` | Evaluar modelos (silhouette score, etc.) |

---

**Volver al [índice](README.md)**
