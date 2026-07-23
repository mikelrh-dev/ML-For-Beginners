# Glosario de NLP — Procesamiento de Lenguaje Natural

## Términos fundamentales

### NLP (Natural Language Processing)
Procesamiento de lenguaje natural. Campo de la IA que estudia cómo las computadoras pueden entender, interpretar y generar lenguaje humano.

### Tokenización
Dividir un texto en unidades más pequeñas llamadas **tokens** (generalmente palabras).

```
"Hola, ¿cómo estás?" → ["Hola", ",", "¿", "cómo", "estás", "?"]
```

### Stemming (Raíz de palabra)
Reducir una palabra a su raíz eliminando prefijos y sufijos.

```
"corriendo" → "corr"
"correr" → "corr"
"corrió" → "corr"
```

### Lematización
Reducir una palabra a su forma base (lema), respetando la gramática.

```
"corriendo" → "correr"
"correr" → "correr"
"corrió" → "correr"
```

**Diferencia con stemming:** Lematización es más inteligente — respeta el idioma.

### Stop Words
Palabras comunes que no aportan significado: "el", "la", "de", "en", "que", "es", "un", "una".

Se eliminan porque ralentizan el análisis sin cambiar el sentimiento.

### POS Tagging (Part-of-Speech)
Etiquetar cada palabra según su función gramatical:

```
"El gato come pescado"
El → artículo
gato → sustantivo
come → verbo
pescado → sustantivo
```

### Parsing
Analizar la estructura gramatical de una oración para entender qué palabras están relacionadas.

### N-grams
Secuencias de N palabras consecutivas:

```
"El gato come pescado"
Unigram: ["El", "gato", "come", "paccado"]
Bigram: ["El gato", "gato come", "come pescado"]
Trigram: ["El gato come", "gato come paccado"]
```

## Sentimiento

### Análisis de sentimiento
Determinar si un texto es **positivo**, **negativo** o **neutral**.

### Polaridad
Mide qué tan positivo o negativo es un texto:
- **-1.0** = muy negativo
- **0.0** = neutral
- **1.0** = muy positivo

### Subjetividad
Mide qué tan objetivo o subjetivo es un texto:
- **0.0** = muy objetivo (hechos)
- **1.0** = muy subjetivo (opiniones)

### VADER
Herramienta de NLTK para análisis de sentimiento. Especialmente buena para redes sociales (maneja emojis, mayúsculas, signos de exclamación).

## Word Embeddings

### Word Embedding
Representación numérica de una palabra en un espacio vectorial. Palabras con significado similar están cerca.

### Vector
Lista de números que codifica el significado de una palabra.

```
"gato" → [0.2, 0.8, 0.1, -0.3, ...]
```

### word2vec
Algoritmo de Google (2013) que crea embeddings entrenando una red neuronal para predecir palabras vecinas.

### GloVe
Algoritmo de Stanford (2014) que crea embeddings basándose en cuántas veces aparecen palabras juntas.

### BERT
Modelo de Google (2018) que crea embeddings **contextuales** — un vector diferente para cada contexto.

## Herramientas de Python

| Herramienta | Para qué sirve |
|-------------|----------------|
| **NLTK** | Tokenización, stemming, sentimiento, stop words |
| **TextBlob** | API simple para NLP (wrapper sobre NLTK) — sentimiento, POS tagging |
| **deep-translator** | Traducción automática (usa Google Translate) |
| **spaCy** | NLP rápido y eficiente para producción |
| **Gensim** | Word embeddings, topic modeling |
| **scikit-learn** | TF-IDF, vectorización de texto |

## Métricas

### Accuracy (Precisión)
Porcentaje de predicciones correctas sobre el total.

### Precision
De todos los que dije "positivo", ¿cuántos realmente lo son?

### Recall (Exhaustividad)
De todos los que realmente son positivos, ¿cuántos detecté?

### F1-Score
Promedio armónico de precision y recall. Balance entre ambos.

## Fórmulas importantes

### TF-IDF
```
TF-IDF = TF × log(N / DF)
```
- **TF**: frecuencia del término en el documento
- **DF**: en cuántos documentos aparece
- **N**: total de documentos

### Polaridad de sentimiento
```
polaridad = (positivas - negativas) / total_palabras
```

### deep-translator
Librería Python para traducción automática. Reemplaza a `TextBlob.translate()` que fue eliminada en v0.20+. Usa Google Translate, Microsoft Translator, etc.

```python
from deep_translator import GoogleTranslator
GoogleTranslator(source='en', target='es').translate("Hello")
```

### Compound Score (VADER)
Score único de -1 a +1 que resume el sentimiento.
- **< -0.05** = negativo
- **-0.05 a 0.05** = neutral
- **> 0.05** = positivo

### Feature Engineering
Crear nuevas variables (features) a partir de datos crudos para mejorar el modelo.

Ejemplos:
- Dirección completa → "Ciudad, País"
- Tags de texto → columnas binarias 0/1
- Texto → score de sentimiento numérico

### One-Hot Encoding
Convertir categorías en columnas binarias (0 o 1).

```
Tags: "Leisure trip, Couple"
→ Leisure_trip: 1, Couple: 1, Solo_traveler: 0, Business_trip: 0, ...
```

### groupby().transform()
Calcular una métrica por grupo y asignarla a cada fila del grupo.

```python
# Cuántas reseñas tiene cada hotel → asignar a cada fila de ese hotel
df['Total_Reviews'] = df.groupby('Hotel_Name')['Hotel_Name'].transform('count')
```

### Correlation with Target
Medir cuánto una feature se relaciona con la variable objetivo.
```python
df['Positive_Sentiment'].corr(df['Reviewer_Score'])  # ~ +0.4
df['Negative_Sentiment'].corr(df['Reviewer_Score'])  # ~ -0.5
```

### Checkpoint
Guardar estado intermedio de un procesamiento largo para no repetirlo si falla algo después.

### Skew (Sesgo de distribución)
Distribución no simétrica.
- **Right skew (positivo)**: cola a la derecha, media > mediana
- **Left skew (negativo)**: cola a la izquierda, media < mediana

---

**Volver al [índice de NLP](README.md)**
