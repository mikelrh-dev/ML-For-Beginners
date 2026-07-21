# 5. Clustering — Agrupando datos sin etiquetas

## ¿Qué es clustering?

Clustering es **aprendizaje no supervisado**: encontramos patrones en datos que no tienen etiquetas.

**La diferencia clave con clasificación:**
- **Clasificación** (sección 4): tenés etiquetas (gato, perro, pájaro), el modelo aprende a predecirlas
- **Clustering** (esta sección): NO tenés etiquetas, el modelo encuentra grupos por sí solo

**Analogía simple:** Es como ordenar una pila de ropa sin etiquetas 🧦👕👖🩲 — no sabés quién es quién, pero agrupás por similitud (tallas, colores, tipo de prenda).

## ¿Para qué sirve clustering?

| Uso real | Ejemplo |
|----------|---------|
| Segmentación de mercado | ¿Qué tipos de clientes compran qué productos? |
| Detección de anomalías | ¿Qué transacciones son sospechosas? |
| Análisis de imágenes | ¿Qué caras son similares? |
| Organización de documentos | ¿Qué artículos hablan del mismo tema? |
| Recomendaciones | "Clientes similares a vos también compraron..." |
| Detección de fraude | ¿Qué tarjetas de crédito se comportan diferente? |

## Conceptos clave

| Concepto | Definición | Analogía |
|----------|------------|----------|
| **Cluster** | Grupo de puntos similares entre sí | Una mesa en un restaurante |
| **Centroid** | Punto central de un cluster (en K-Means) | El centro de masa de un grupo |
| **Silhouette Score** | Mide qué tan bien separados están los clusters (-1 a 1) | Qué tan claros están los colores en un batido |
| **WCSS** | Within-Cluster Sum of Squares — qué tan compactos son los clusters | Qué apretada está la gente en una mesa |
| **Método del codo** | Técnica para elegir el número óptimo de clusters | ¿Cuántas mesas necesito en el restaurante? |
| **Outliers** | Valores atípicos que distorsionan los resultados | La persona de 80 años en un grupo de 20 |

## Algoritmos principales

| Algoritmo | Cuándo usarlo | Ventaja | Desventaja |
|-----------|---------------|---------|------------|
| **K-Means** | Propósito general, clusters esféricos | Rápido, simple | Hay que elegir k manualmente |
| **DBSCAN** | Clusters de forma irregular, ruido | Detecta outliers automáticamente | Sensible a parámetros |
| **Gaussian Mixture** | Clusters superpuestos | Probabilístico, flexible | Más complejo |
| **Hierarchical** | Cuando querés ver la jerarquía | Visual (dendrograma) | Lento con muchos datos |

## Lecciones de esta sección

1. **[Visualización de datos](1-Visualize.md)** — Explorar el dataset de música nigeriana antes de clustering
2. **[K-Means](2-K-Means.md)** — Aplicar el algoritmo más común y evaluar resultados

## Dato curioso

El análisis de clusters originó en Antropología y Psicología en los años 1930. Se usaba para clasificar culturas y comportamientos humanos antes de que existiera la computación.

---

**Siguiente sección:** [6-NLP](../6-NLP/README.md) — Procesamiento de lenguaje natural
