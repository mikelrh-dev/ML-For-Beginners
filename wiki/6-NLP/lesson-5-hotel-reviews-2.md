# Lección 5: Hotel Reviews II — Análisis de Sentimiento con VADER

> **Convertir texto en números para que una máquina lo entienda.**

---

## ¿Qué aprendemos aquí?

En la lección anterior exploramos 515K reseñas. Ahora las transformamos en datos listos para ML:

1. **Simplificar direcciones** — Unificar direcciones largas en `Ciudad, País`
2. **Limpiar columnas** — Eliminar ruido y recalcular métricas consistentes
3. **Convertir tags** — Texto como "Leisure trip, Couple" → columnas binarias (0/1)
4. **Análisis de sentimiento** — VADER: texto → puntaje -1 a +1

---

## El pipeline

```
Hotel_Reviews.csv (original)
       ↓
Simplificar direcciones
       ↓
Eliminar columnas redundantes
       ↓
Recalcular métricas por hotel
       ↓
Tags → columnas binarias
       ↓
Hotel_Reviews_Filtered.csv (checkpoint)
       ↓
Eliminar stop words
       ↓
VADER: Positive_Review → Positive_Sentiment (-1 a +1)
       ↓
VADER: Negative_Review → Negative_Sentiment (-1 a +1)
       ↓
Hotel_Reviews_NLP.csv (listo para ML)
```

---

## El código: paso a paso

### 1. Simplificar direcciones

```python
def replace_address(row):
    if "Netherlands" in row["Hotel_Address"]:
        return "Amsterdam, Netherlands"
    elif "Barcelona" in row["Hotel_Address"]:
        return "Barcelona, Spain"
    # ... más ciudades

df["Hotel_Address"] = df.apply(replace_address, axis=1)
```

**¿Por qué?** La dirección original es `Savva Street 12 Amsterdam Netherlands`. Para agrupar por ciudad, necesitamos formato uniforme: `Amsterdam, Netherlands`.

---

### 2. Eliminar columnas y recalcular

```python
# Eliminar columnas que no sirven
df.drop(["Additional_Number_of_Scoring",
         "Review_Total_Negative_Word_Counts",
         "Review_Total_Positive_Word_Counts",
         "days_since_review",
         "Total_Number_of_Reviews_Reviewer_Has_Given"], axis=1, inplace=True)

# Recalcular: agrupar por hotel y contar
df['Total_Number_of_Reviews'] = df.groupby('Hotel_Name')['Hotel_Name'].transform('count')

# Recalcular: promedio de score por hotel
df['Average_Score'] = round(df.groupby('Hotel_Name')['Reviewer_Score'].transform('mean'), 1)
```

**¿Por qué?** Las columnas originales pueden estar desactualizadas. Recalculamos con `groupby().transform()` para que cada fila del hotel tenga el mismo valor correcto.

---

### 3. Tags → columnas binarias

```python
# Limpiar: "[ 'Leisure trip', 'Couple' ]" → "Leisure trip, Couple"
df.Tags = df.Tags.str.strip("['").str.replace("', '", ",", regex=False)

# One-hot encoding manual
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
# ... más tags
```

**¿Por qué?** Las máquinas no entienden "Leisure trip, Couple". Entienden `Leisure_trip=1, Couple=1, Solo_traveler=0`. Esto se llama **one-hot encoding**.

---

### 4. Guardar checkpoint (¡importante!)

```python
df.to_csv('../data/Hotel_Reviews_Filtered.csv', index=False)
```

**¿Por qué?** El siguiente paso (VADER) tarda minutos en 515K filas. Si falla, no quieres repetir lo anterior.

---

### 5. VADER: análisis de sentimiento

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('stopwords')

from nltk.corpus import stopwords

# Stop words = palabras sin significado para sentimiento
cache = set(stopwords.words("english"))

def remove_stopwords(review):
    return " ".join([w for w in review.split() if w not in cache])

df.Negative_Review = df.Negative_Review.apply(remove_stopwords)
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)

# Analizador VADER
vader = SentimentIntensityAnalyzer()

def calc_sentiment(review):
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader.polarity_scores(review)["compound"]

df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
```

---

## ¿Qué es VADER?

**VADER** = Valence Aware Dictionary and sEntiment Reasoner

Es un analizador **basado en reglas** (no ML) con un diccionario de palabras:

| Palabra | Score |
|---------|-------|
| `amazing` | +0.7 |
| `terrible` | -0.8 |
| `okay` | +0.1 |

**Ventajas sobre TextBlob:**
- Entiende **mayúsculas**: `AMAZING` > `amazing`
- Detecta **puntuación**: `good!!!` > `good`
- Maneja **emojis**: 😊 = positivo, 😡 = negativo
- **Rápido** para 515K reseñas

---

## ¿Por qué stop words?

Palabras como `the`, `is`, `at`, `which` no aportan sentimiento.

```python
"The room was amazing"  →  "room amazing"
"was not good"  →  "not good"  ← ¡cuidado con negaciones!
```

VADER maneja negaciones mejor que otros, pero **eliminar stop words puede romper "not bad" → "bad"**. En práctica, VADER es robusto.

---

## Verificación: correlación con score real

```python
print(df['Negative_Sentiment'].corr(df['Reviewer_Score']))  # esperado: negativo
print(df['Positive_Sentiment'].corr(df['Reviewer_Score']))  # esperado: positivo
```

Si VADER funciona:
- **Más sentimiento negativo** → **score menor** (correlación ~ -0.4 a -0.6)
- **Más sentimiento positivo** → **score mayor** (correlación ~ +0.3 a +0.5)

---

## Dataset final

| Columna | Tipo | Qué es |
|---------|------|--------|
| `Hotel_Name` | texto | Nombre hotel |
| `Hotel_Address` | texto | Ciudad, País |
| `Total_Number_of_Reviews` | int | Reseñas totales del hotel |
| `Average_Score` | float | Score promedio del hotel |
| `Reviewer_Score` | float | **Target**: score de esta reseña |
| `Negative_Sentiment` | float | VADER score reseña negativa (-1 a +1) |
| `Positive_Sentiment` | float | VADER score reseña positiva (-1 a +1) |
| `Reviewer_Nationality` | texto | País revisor |
| `Leisure_trip` ... `With_a_pet` | int (0/1) | Tags binarios |
| `Negative_Review` | texto | Texto original |
| `Positive_Review` | texto | Texto original |

---

## Conceptos clave

| Concepto | Definición |
|----------|------------|
| **VADER** | Analizador de sentimiento basado en reglas/diccionario |
| **Compound score** | Score único -1 a +1 (negativo/positivo/neutral) |
| **Stop words** | Palabras comunes sin valor semántico ("the", "is") |
| **One-hot encoding** | Convertir categoría en columna 0/1 |
| **groupby().transform()** | Calcular métrica por grupo y asignar a cada fila |
| **Feature engineering** | Crear features útiles a partir de datos crudos |
| **Checkpoint** | Guardar estado intermedio para no repetir trabajo |

---

## Errores comunes

1. **No usar `transform('count')` con columna específica** → `df.groupby().transform('count')` devuelve TODAS las columnas
2. **Olvidar `nltk.download('stopwords')`** → LookupError
3. **No guardar checkpoint** → repetir procesamiento si falla VADER
4. **Eliminar stop words sin cuidado** → romper "not bad" → "bad" (VADER lo mitiga)

---

## Notas técnicas

- **Tiempo VADER**: ~3-5 min en 515K filas. Paciencia.
- **Memoria**: Stop words + VADER usan RAM extra. Si falla, procesa en chunks.
- **VADER no es perfecto**: Ironía, sarcasmo, contexto cultural → errores. Es regla heurística, no ML.

---

## Siguiente paso

Con `Hotel_Reviews_NLP.csv` tienes features numéricos listos para:

- **Clustering** → agrupar hoteles por tipo de cliente
- **Clasificación** → predecir `Reviewer_Score` a partir de sentimiento + tags
- **Recomendación** → sugerir hoteles según preferencias

Las técnicas las viste en secciones anteriores (Clustering, Clasificación). Ahora con datos reales.

---