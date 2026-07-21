# 5. Clustering — Agrupando datos sin etiquetas

## ¿Qué aprendemos aquí?

Clustering es **aprendizaje no supervisado**: encontramos patrones en datos que no tienen etiquetas. Es como ordenar una pila de ropa sin etiquetas — agrupás por similitud.

**Diferencia con clasificación:**
- **Clasificación** (sección 4): tenés etiquetas, el modelo aprende a predecirlas
- **Clustering** (esta sección): NO tenés etiquetas, el modelo encuentra grupos por sí solo

## Ejemplos reales

| Uso | Ejemplo |
|-----|---------|
| Segmentación de mercado | ¿Qué tipos de clientes compran qué productos? |
| Detección de anomalías | ¿Qué transacciones son sospechosas? |
| Análisis de imágenes | ¿Qué caras son similares? |
| Organización de documentos | ¿Qué artículos hablan del mismo tema? |

## Conceptos clave

| Concepto | Definición |
|----------|------------|
| **Cluster** | Grupo de puntos similares entre sí |
| **Centroid** | Punto central de un cluster (en K-Means) |
| **Silhouette Score** | Mide qué tan bien separados están los clusters (-1 a 1) |
| **WCSS** | Within-Cluster Sum of Squares — qué tan compactos son los clusters |
| **Método del codo** | Técnica para elegir el número óptimo de clusters |

## Algoritmos principales

| Algoritmo | Cuándo usarlo |
|-----------|---------------|
| **K-Means** | Propósito general, clusters esféricos |
| **DBSCAN** | Clusters de forma irregular, ruido |
| **Gaussian Mixture** | Clusters superpuestos, distribución normal |
| **Hierarchical** | Cuando querés ver la jerarquía de clusters |

## Lecciones

1. [Visualización de datos](1-Visualize/README.md) — Explorar el dataset de música nigeriana
2. [K-Means](2-K-Means/README.md) — Aplicar el algoritmo más común de clustering

## Dato curioso

El análisis de clusters originó en Antropología y Psicología en los años 1930. Se usaba para clasificar culturas y comportamientos humanos antes de que existiera la computación.

---

**Siguiente sección:** [6-NLP](../6-NLP/README.md) — Procesamiento de lenguaje natural
