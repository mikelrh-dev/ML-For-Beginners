# Análisis de sentimiento con reseñas de hoteles — Procesamiento de datos

En esta sección usarás las técnicas de lecciones anteriores para hacer análisis exploratorio de un dataset grande. Aprenderás:

- Cómo eliminar columnas innecesarias
- Cómo calcular nuevos datos basados en columnas existentes
- Cómo guardar el dataset para el desafío final

## [Quiz pre-lección](https://ff-quizzes.netlify.app/en/ml/)

### Introducción

Hasta ahora aprendiste que los datos de texto son diferentes a los numéricos. Si fue escrito o hablado por un humano, puede analizarse para encontrar patrones, frecuencias, sentimiento y significado.

Esta lección usa un dataset real: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** con licencia [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).

### Preparación

Necesitás:
- Python 3 con notebooks
- pandas
- NLTK ([instalar localmente](https://www.nltk.org/install.html))
- El dataset de Kaggle (~230 MB descomprimido)

## Análisis exploratorio de datos

El dataset incluye reseñas de 1493 hoteles diferentes en 6 ciudades. Podés descubrir:

- ¿Cuáles son las palabras y frases más frecuentes?
- ¿Los tags oficiales correlacionan con las puntuaciones?
- ¿Los scores de sentimiento de NLTK "coinciden" con la puntuación del revisor?

### Estructura del dataset

**Columnas de hotel:**
- `Hotel_Name`, `Hotel_Address`, `lat`, `lng`

**Columnas meta-reseña:**
- `Average_Score` — promedio basado en el último comentario del último año
- `Total_Number_of_Reviews` — total de reseñas del hotel
- `Additional_Number_of_Scoring` — puntuaciones sin reseña escrita

**Columnas de reseña:**
- `Reviewer_Score` — puntuación numérica (2.5 a 10)
- `Negative_Review` — reseña negativa ("No Negative" si está vacía)
- `Positive_Review` — reseña positiva ("No Positive" si está vacía)
- `Tags` — descriptores como tipo de viaje, tipo de huésped, etc.

**Columnas de revisor:**
- `Reviewer_Nationality` — nacionalidad del revisor
- `Total_Number_of_Reviews_Reviewer_Has_Given` — total de reseñas del revisor

### Ejemplo

| Avg Score | Reviews | Score | Negative Review | Positive Review |
|-----------|---------|-------|-----------------|-----------------|
| 7.8 | 1945 | 2.5 | "This is currently not a hotel but a construction site..." | "Nothing Terrible place Stay away" |

### Tags

Los tags no son estandarizados: un hotel tiene "Single room" y otro "Deluxe Single Room". Pero podemos usar NLP para medir frecuencias de términos como *Solo*, *Business Traveller*, o *Family with young kids*.

## Ejercicio: exploración de datos

### Cargar datos

```python
import pandas as pd
import time

print("Cargando datos...")
start = time.time()
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Cargado en " + str(round(end - start, 2)) + " segundos")
```

### Operaciones con el DataFrame

1. **Forma del dataset:**
   ```python
   print("La forma de los datos (filas, columnas) es " + str(df.shape))
   # (515738, 17)
   ```

2. **Frecuencia de nacionalidades:**
   ```python
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("Hay " + str(nationality_freq.size) + " nacionalidades diferentes")
   # United Kingdom: 245246
   # United States: 35437
   # Australia: 21686
   ```

3. **Hotel más reseñado por nacionalidad:**
   ```python
   for nat in nationality_freq[:10].index:
       nat_df = df[df["Reviewer_Nationality"] == nat]   
       freq = nat_df["Hotel_Name"].value_counts()
       print("El hotel más reseñado por " + str(nat).strip() + " fue " + str(freq.index[0]))
   ```

4. **Calcular promedio propio:**
   ```python
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   ```

5. **Contar reseñas vacías:**
   ```python
   no_negative = sum(df.Negative_Review == "No Negative")    # 127,890
   no_positive = sum(df["Positive_Review"] == "No Positive") # 35,946
   ```

---

## 🚀Desafío

Esta lección demuestra lo importante que es **entender tus datos** antes de operar sobre ellos. Los datos de texto en particular requieren escrutinio cuidadoso.

## [Quiz post-lección](https://ff-quizzes.netlify.app/en/ml/)

## Tarea

[NLTK](assignment.md)
