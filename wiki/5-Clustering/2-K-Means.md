# Lección 2: K-Means Clustering

## ¿Qué aprendemos aquí?

K-Means es el algoritmo de clustering más popular. Funciona como ordenar personas por estatura: primero elegís cuántos grupos querés, y el algoritmo va agrupando por similitud.

## ¿Cómo funciona K-Means?

```
1. Elegís k (número de clusters)
2. El algoritmo coloca k centroids aleatoriamente
3. Asigna cada punto al centroid más cercano
4. Recalcula los centroids como el promedio
5. Repite hasta que se estabilicen
```

**Visualización:** Imaginá un diagrama de Voronoi — cada región es un cluster, y cada centroid es el "centro" de su región.

## Conceptos clave

### Silhouette Score
Mide qué tan bien separados están los clusters:
- **1**: cluster denso y bien separado
- **0**: clusters superpuestos
- **-1**: puntos en cluster equivocado

Nuestro resultado: ~0.53 (mediocre, indica superposición)

### Método del codo (Elbow Method)
Técnica para elegir k:
1. Probás k de 1 a 10
2. Graficás la inercia (WCSS)
3. El "codo" indica el k óptimo

En nuestro caso, el codo sugiere k=3 (coincide con los 3 géneros)

### Inercia (WCSS)
Within-Cluster Sum of Squares — mide qué tan compactos son los clusters. Menor = mejor, pero siempre decrece con más clusters.

## ¿Qué salió mal?

La precisión fue baja. ¿Por qué?

| Problema | Explicación |
|----------|-------------|
| **Poca correlación** | Las características no están muy relacionadas |
| **Outliers** | Valores extremos distorsionan los centroids |
| **Superposición** | Los géneros musicales se mezclan en estas dimensiones |

## Lección importante

> **No todos los datasets son buenos candidatos para K-Means.** Si los datos no tienen estructura de cluster natural, el algoritmo va a forzar agrupaciones artificiales.

## ¿Qué podrías mejorar?

1. **Escalar los datos** (StandardScaler) — para que todas las columnas tengan el mismo peso
2. **Usar otras columnas** — quizás instrumentalness o speechiness ayuden
3. **Probar otro algoritmo** — DBSCAN o Gaussian Mixture pueden funcionar mejor
4. **Limpiar más outliers** — reducir el ruido

## Analogía

Es como intentar separar colores en un cuadro que tiene mucho blur. Si los colores están muy mezclados, no importa cuántas veces reorganices — no vas a lograr separación nítida.

---

**Siguiente sección:** [6-NLP](../../6-NLP/README.md) — Procesamiento de lenguaje natural
