# Tareas y técnicas comunes de NLP

Para la mayoría de las tareas de *procesamiento de lenguaje natural*, el texto a procesar debe ser descompuesto, examinado y los resultados almacenados o cruzados con reglas y conjuntos de datos.

## [Quiz pre-lección](https://ff-quizzes.netlify.app/en/ml/)

Descubrí las técnicas comunes usadas en procesamiento de texto. Combinadas con machine learning, te ayudan a analizar grandes cantidades de texto eficientemente.

## Tareas comunes de NLP

### Tokenización

Lo primero que hacen los algoritmos de NLP es dividir el texto en **tokens** (palabras). Suena simple, pero hay que considerar puntuación y delimitadores de diferentes idiomas.

![tokenization](images/tokenization.png)
> Tokenizando una oración de **Pride and Prejudice**. Infografía de [Jen Looper](https://twitter.com/jenlooper)

### Embeddings (Incrustaciones)

Los [word embeddings](https://wikipedia.org/wiki/Word_embedding) convierten texto a números de forma que palabras con significado similar se agrupen juntas.

![word embeddings](images/embedding.png)
> Embeddings para una oración de **Pride and Prejudice**. Infografía de [Jen Looper](https://twitter.com/jenlooper)

✅ Probá [esta herramienta](https://projector.tensorflow.org/) para experimentar con word embeddings.

### Parsing y etiquetado POS

Cada palabra puede ser etiquetada como parte del discurso: sustantivo, verbo o adjetivo. La oración `the quick red fox jumped over the lazy brown dog` se etiquetaría como: fox = sustantivo, jumped = verbo.

![parsing](images/parse.png)
> Parseando una oración de **Pride and Prejudice**. Infografía de [Jen Looper](https://twitter.com/jenlooper)

Parsing reconoce qué palabras están relacionadas en una oración.

### Frecuencias de palabras y frases

Construir un diccionario de cada palabra o frase de interés y cuántas veces aparece.

La frase `the quick red fox jumped over the lazy brown dog` tiene frecuencia 2 para "the".

### N-grams

Un texto puede dividirse en secuencias de palabras de longitud fija: unigram (1 palabra), bigram (2), trigram (3), o n-gram (n).

Ejemplo con n-gram de 2:
1. the quick 
2. quick red 
3. red fox
4. fox jumped 

![n-grams sliding window](images/n-grams.gif)
> N-gram de 3: Infografía de [Jen Looper](https://twitter.com/jenlooper)

### Extracción de frases sustantivas

Identificar el sustantivo principal de una oración. En `the quick red fox jumped over the lazy brown dog` hay 2 frases sustantivas: **quick red fox** y **lazy brown dog**.

### Análisis de sentimiento

Una oración puede analizarse para saber qué tan *positiva* o *negativa* es:
- **Polaridad**: de -1.0 (negativo) a 1.0 (positivo)
- **Subjetividad**: de 0.0 (objetivo) a 1.0 (subjetivo)

### Lematización

Una *lema* es la raíz de un conjunto de palabras. Por ejemplo: *flew*, *flies*, *flying* tienen como lema el verbo *fly*.

### WordNet

[WordNet](https://wordnet.princeton.edu/) es una base de datos de palabras, sinónimos, antónimos y muchos detalles para cada palabra en varios idiomas. Es muy útil para construir traducciones y correctores ortográficos.

## Librerías de NLP

No tenés que construir todo desde cero — hay excelentes librerías de Python disponibles.

### Usando TextBlob

TextBlob "se apoya en los gigantes de [NLTK](https://nltk.org) y [pattern](https://github.com/clips/pattern)" y tiene ML integrado en su API.

Para identificar *frases sustantivas*, TextBlob ofrece varias opciones de extractores:

```python
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

extractor = ConllExtractor()

user_input = input("> ")
user_input_blob = TextBlob(user_input, np_extractor=extractor)
np = user_input_blob.noun_phrases
```

### Desafío: mejorar tu bot con NLP

Ahora harás a Marvin más simpático analizando el sentimiento de la entrada y respondiendo acorde.

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Ejemplo de salida:
```
Hello, I am Marvin, the friendly robot.
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> bye
It was nice talking to you, goodbye!
```

Una solución posible está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

---

## 🚀Desafío

Implementá el bot de la verificación de conocimiento y probalo con un amigo. ¿Puede engañarlo?

## [Quiz post-lección](https://ff-quizzes.netlify.app/en/ml/)

## Tarea

[Hacer que un bot responda](assignment.md)
