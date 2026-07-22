# Traducción y análisis de sentimiento con ML

En las lecciones anteriores aprendiste a construir un bot básico usando `TextBlob`. Otro desafío importante en lingüística computacional es la _traducción_ precisa de oraciones de un idioma a otro.

## [Quiz pre-lección](https://ff-quizzes.netlify.app/en/ml/)

## Traducción

La traducción es un problema muy difícil porque hay miles de idiomas con reglas gramaticales diferentes. Un enfoque es convertir las reglas gramaticales a una estructura independiente del idioma y luego traducir de vuelta.

### Ejemplo: inglés a irlandés

En inglés: *I feel happy* = sujeto + verbo + adjetivo

En irlandés: *Tá athas orm* = verbo + adjetivo + sujeto (literalmente: "feliz está sobre mí")

Un programa de traducción naivo podría traducir palabra por palabra y producir algo sin sentido.

> 🎥 [Video sobre tradiciones lingüísticas irlandesas](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Enfoques de Machine Learning

Otro enfoque es ignorar el significado de las palabras y usar ML para detectar patrones. Si tenés mucho texto (*corpus*) en ambos idiomas, el modelo puede aprender traducciones idiomáticas.

Por ejemplo, `I have no money` traducido literalmente al francés sería `Je n'ai pas de monnaie` (monnaie = cambio suelto). Un humano diría `Je n'ai pas d'argent` (dinero).

### Ejercicio: traducción con TextBlob

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))
```

Resultado: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!"

## Análisis de sentimiento

Otra área donde ML funciona bien es el análisis de sentimiento. Un enfoque sin ML es identificar palabras "positivas" y "negativas" y calcular el total.

### El problema del sarcasmo

La oración `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` es sarcástica y negativa, pero un algoritmo simple detecta 'great', 'wonderful', 'glad' como positivas.

### Enfoques de ML

El enfoque ML es recopilar textos con puntuación y opinión escrita, y que el modelo aprenda patrones. Por ejemplo, las reseñas positivas de películas tienden a tener 'Oscar worthy' más que las negativas.

### Ejercicio: sentimiento con Pride and Prejudice

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " tiene sentimiento " + str(sentiment1))
print(quote2 + " tiene sentimiento " + str(sentiment2))
```

Resultado:
```
Sentiment(polarity=0.21, subjectivity=0.27)  ← primera oración
Sentiment(polarity=0.7, subjectivity=0.8)    ← última oración
```

## Desafío

Determiná, usando polaridad de sentimiento, si *Pride and Prejudice* tiene más oraciones absolutamente positivas que negativas.

**Pasos:**
1. Descargá una [copia de Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) como archivo .txt
2. Abrilo en Python y extraé el contenido como string
3. Creá un TextBlob con el string del libro
4. Analizá cada oración en un loop
5. Imprimí las oraciones positivas y negativas por separado

Una solución está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)

---

## 🚀Desafío

¿Puedes hacer a Marvin aún mejor extrayendo otras características de la entrada del usuario?

## [Quiz post-lección](https://ff-quizzes.netlify.app/en/ml/)

## Tarea

[Licencia poética](assignment.md)
