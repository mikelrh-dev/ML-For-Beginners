# Lección 4: Hotel Reviews I — Análisis Exploratorio de Datos (EDA)

> **Antes de modelar, explora. Saltar este paso es la causa #1 de modelos que fallan.**

---

## ¿Qué aprendemos aquí?

Esta lección enseña a **entender un dataset real de 515,000 reseñas** antes de tocar cualquier modelo de ML.

Vas a aprender:
1. Cómo cargar un CSV grande sin romper la memoria
2. Qué columnas importan y cuáles no
3. Cómo detectar problemas (nulos, inconsistencias)
4. Cómo identificar la variable objetivo (puntuación del revisor)

---

## El código

```python
import pandas as pd
import time

# 1. Cargar
df = pd.read_csv('../data/Hotel_Reviews.csv')

# 2. Explorar forma y columnas
print(df.shape)
print(df.columns.tolist())
df.info()
df.describe()

# 3. Ver valores únicos por columna
for col in df.columns:
    print(f"{col}: {df[col].nunique()} únicos")

# 4. Valores faltantes
df.isnull().sum()

# 5. Distribución de la puntuación (target)
df['Reviewer_Score'].hist(bins=10)
```

---

## Desglose paso a paso

### 1. Cargar el dataset

```python
import pandas as pd
import time

start = time.time()
df = pd.read_csv('../data/Hotel_Reviews.csv')
print(f"Carga: {round(time.time() - start, 2)} seg")
```

**¿Por qué medir el tiempo?** Con 515K filas y ~240 MB, la carga toma segundos. En datasets reales de GB, minutos.

---

### 2. Forma y columnas

```python
df.shape          # (515738, 17)
df.columns.tolist()
```

Salida típica:
```
['Hotel_Address', 'Additional_Number_of_Scoring',
 'Review_Date', 'Average_Score', 'Hotel_Name',
 'Reviewer_Nationality', 'Negative_Review',
 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts',
 'Reviewer_Score', 'Total_Number_of_Reviews',
 'Total_Number_of_Reviews_Reviewer_Has_Given',
 'Reviewer_Score', 'Tags', 'days_since_review',
 'lat', 'lng']
```

---

### 3. `df.info()` — Tu mejor amigo

```python
df.info()
```

Te dice:
- **Tipo de dato** de cada columna (int64, float64, object)
- **Valores no nulos** — detecta missing values
- **Memoria usada** — importante para datasets grandes

---

### 4. `df.describe()` — Estadísticas rápidas

```python
df.describe()
```

Solo columnas numéricas. Te da: count, mean, std, min, 25%, 50%, 75%, max.

Útil para detectar outliers (ej. `Reviewer_Score` = 10 en escala 1-10 es raro).

---

### 5. Valores únicos por columna

```python
for col in df.columns:
    print(f"{col:40s}: {df[col].nunique():>6} únicos")
```

**Qué buscas:**
- Columnas con **1 único** = constante, eliminar
- Columnas con **muchos únicos** (ej. Hotel_Address) = necesita limpieza
- `Reviewer_Score` debería tener 10 valores (1.0 a 10.0)

---

### 6. Valores faltantes

```python
df.isnull().sum()
```

Si hay nulos:
- **Pocos** → `df.dropna()` o imputar
- **Muchos** → investigar por qué faltan (¿error de recolección?)

---

### 7. Distribución del target

```python
import matplotlib.pyplot as plt

df['Reviewer_Score'].hist(bins=10, figsize=(8,4))
plt.xlabel('Puntuación (1-10)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Reviewer_Score')
plt.show()
```

**Qué esperas ver:** Campana alrededor de 7-8 (la gente suele puntuar alto). Si ves picos raros, investiga.

---

## ¿Por qué esto?

| Paso | Pregunta que responde |
|------|----------------------|
| `shape` | ¿Cuántos datos tengo? |
| `info()` | ¿Qué tipo de datos? ¿Hay nulos? |
| `describe()` | ¿Rangos normales? ¿Outliers? |
| `nunique()` | ¿Hay columnas constantes? ¿Texto que limpiar? |
| `isnull().sum()` | ¿Datos incompletos? |
| `hist()` | ¿Target balanceado? ¿Sesgo? |

**Regla de oro:** No modeles lo que no entiendes. Un EDA de 10 minutos te ahorra días de debugging.

---

## Conceptos clave

| Concepto | Definición |
|----------|------------|
| **EDA** | Exploratory Data Analysis — entender datos antes de modelar |
| **Target / Variable objetivo** | Lo que quieres predecir (`Reviewer_Score`) |
| **Outlier** | Valor extremo que distorsiona el modelo |
| **Missing value** | Dato faltante | Dato que no se registró (NaN) |
| **Feature** | Columna de entrada para el modelo |
| **Skew / Sesgo** | Distribución no simétrica |

---

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| Saltar EDA | Modelar basura → resultados basura |
| No ver `info()` | No detectar nulos hasta que el modelo falla |
| No graficar target | No saber si está desbalanceado |
| Asumir que los datos están limpios | `Hotel_Address` tiene 5000 variantes únicas |

---

## Siguiente paso

[**Lección 5: Hotel Reviews 2 — Sentimiento**](lesson-5-hotel-reviews-2.md) → Limpiar datos, convertir tags, calcular sentimiento con VADER, crear dataset final para ML.