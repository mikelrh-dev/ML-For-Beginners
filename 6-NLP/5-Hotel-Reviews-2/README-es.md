# Análisis de sentimiento con reseñas de hoteles

Ahora que exploraste el dataset en detalle, es hora de filtrar las columnas y usar técnicas de NLP para obtener nuevos insights sobre los hoteles.

## [Quiz pre-lección](https://ff-quizzes.netlify.app/en/ml/)

## Filtrado y análisis de sentimiento

Como probablemente notaste, el dataset tiene algunos problemas. Algunas columnas tienen información inútil, otras parecen incorrectas.

## Ejercicio: más procesamiento de datos

### 1. Procesamiento inicial de columnas

**Eliminar `lat` y `lng`**, y reemplazar `Hotel_Address` con ciudad y país:

```python
def replace_address(row):
    if "Netherlands" in row["Hotel_Address"]:
        return "Amsterdam, Netherlands"
    elif "Barcelona" in row["Hotel_Address"]:
        return "Barcelona, Spain"
    elif "United Kingdom" in row["Hotel_Address"]:
        return "London, United Kingdom"
    elif "Milan" in row["Hotel_Address"]:        
        return "Milan, Italy"
    elif "France" in row["Hotel_Address"]:
        return "Paris, France"
    elif "Vienna" in row["Hotel_Address"]:
        return "Vienna, Austria" 

df["Hotel_Address"] = df.apply(replace_address, axis = 1)
```

Resultado:
| Hotel_Address | Hoteles |
|---------------|---------|
| Amsterdam, Netherlands | 105 |
| Barcelona, Spain | 211 |
| London, United Kingdom | 400 |
| Milan, Italy | 162 |
| Paris, France | 458 |
| Vienna, Austria | 158 |

### 2. Columnas meta-reseña

```python
df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
```

### 3. Columnas de reseña

Eliminar conteos de palabras y fechas, mantener puntuación y texto de reseñas.

### 4. Procesar tags

Los tags son problemáticos porque son listas de texto con orden variable. Usamos NLP para encontrar las frases más comunes:

```python
df.Tags = df.Tags.str.strip("[']")
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

**Tags más comunes:**
| Tag | Cantidad |
|-----|----------|
| Leisure trip | 417,778 |
| Couple | 252,294 |
| Solo traveler | 108,545 |
| Business trip | 82,939 |
| Family with young children | 61,015 |

### Crear columnas de tags

```python
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)
```

### Guardar archivo filtrado

```python
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Análisis de sentimiento

### Eliminar stop words

Las stop words (palabras comunes como "the", "is", "at") no cambian el sentimiento pero ralentizan el análisis:

```python
from nltk.corpus import stopwords

cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Calcular sentimiento con VADER

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader_sentiment = SentimentIntensityAnalyzer()

def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    

df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
```

### Guardar resultado final

```python
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", 
                 "Average_Score", "Reviewer_Score", "Negative_Sentiment", 
                 "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", 
                 "Couple", "Solo_traveler", "Business_trip", "Group", 
                 "Family_with_young_children", "Family_with_older_children", 
                 "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

## Resumen del flujo

1. **Hotel_Reviews.csv** → explorado en la lección anterior
2. **Hotel_Reviews_Filtered.csv** → filtrado en esta lección
3. **Hotel_Reviews_NLP.csv** → con análisis de sentimiento

## Conclusión

Empezaste con un dataset con columnas que no todas podían verificarse. Exploraste, filtraste, convertiste tags, calculaste promedios, agregaste sentimiento y aprendiste sobre procesamiento de texto natural.

---

## 🚀Desafío

Usando el dataset analizado, aplicá clustering u otras estrategias para encontrar patrones alrededor del sentimiento.

## [Quiz post-lección](https://ff-quizzes.netlify.app/en/ml/)

## Tarea

[Probar un dataset diferente](assignment.md)
