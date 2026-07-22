# Lección 2: El Bot Marvin — Análisis de Sentimiento con TextBlob

## ¿Qué aprendemos aquí?

Cómo funciona un chatbot que **entiende emociones** (básicamente). Marvin analiza lo que le decís y responde según el sentimiento de tu texto. No es inteligencia artificial real — es un árbol de decisiones basado en polaridad numérica.

---

## El código

```python
import random
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()

def main():   
    print("Hello, I am Marvin, the friendly robot.")
    print("You can end this conversation at any time by typing 'bye'")    
    print("After typing each answer, press 'enter'")
    print("How are you today?")

    while True:
        user_input = input("> ")

        if user_input.lower() == "bye":            
            break
        else:
            user_input_blob = TextBlob(user_input, np_extractor=extractor)                        
            np = user_input_blob.noun_phrases                                    
            response = ""
            if user_input_blob.polarity <= -0.5:
                response = "Oh dear, that sounds bad. "
            elif user_input_blob.polarity <= 0:
                response = "Hmm, that's not great. "
            elif user_input_blob.polarity <= 0.5:
                response = "Well, that sounds positive. "
            elif user_input_blob.polarity <= 1:
                response = "Wow, that sounds great. "

            if len(np) != 0:
                response = response + "Can you tell me more about " + np[0].pluralize() + "?"
            else:
                response = response + "Can you tell me more?"
            print(response)
    
    print("It was nice talking to you, goodbye!")

main()
```

---

## Desglose paso a paso

### Paso 1: Las importaciones

```python
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()
```

**TextBlob** es como un "cuchillo suizo" del NLP. Viene con herramientas para:
- Analizar sentimiento (¿positivo o negativo?)
- Extraer frases sustantivas (¿de qué hablás?)
- Traducir (en la próxima lección)

**ConllExtractor** es un extractor de frases sustantivas que usa un modelo entrenado con el corpus CoNLL. Es más preciso que el extractor default.

¿Por qué importa? Porque `TextBlob("my cat is happy").noun_phrases` devuelve `["cat"]` — extrae el sustantivo central.

### Paso 2: La presentación

```python
print("Hello, I am Marvin, the friendly robot.")
print("You can end this conversation at any time by typing 'bye'")    
print("After typing each answer, press 'enter'")
print("How are you today?")
```

Cuatro prints simples. Nada mágico aquí — solo está estableciendo las reglas del juego.

### Paso 3: El loop infinito

```python
while True:
    user_input = input("> ")
```

`while True` es un loop que **nunca para solo**. Solo se detiene con `break`.

`input("> ")` hace dos cosas:
1. Muestra un `>` en pantalla
2. Espera a que escribas algo y presiones Enter

**Truco**: En Jupyter, `input()` muestra una caja de texto arriba del notebook. En terminal, aparece en la misma línea.

### Paso 4: La salida

```python
if user_input.lower() == "bye":            
    break
```

- `.lower()` convierte "BYE", "ByE", "bye" → "bye"
- Si coincide, `break` rompe el `while True` y el bot se despide

### Paso 5: El análisis de sentimiento

```python
user_input_blob = TextBlob(user_input, np_extractor=extractor)
np = user_input_blob.noun_phrases
```

**Línea 1**: Crea un objeto TextBlob con tu texto y el extractor de frases sustantivas.

**Línea 2**: Extrae las frases sustantivas. Ejemplos:

| Input | noun_phrases |
|-------|--------------|
| "my cat is happy" | `["cat"]` |
| "I love programming" | `["programming"]` |
| "the weather is terrible today" | `["weather"]` |
| "hello" | `[]` (vacío) |

### Paso 6: El árbol de decisiones de sentimiento

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

**Polarity** es un número de -1 a 1:

```
-1 -------- -0.5 -------- 0 -------- 0.5 -------- 1
  muy mal     mal      neutral     bien      muy bien
```

| Polaridad | Respuesta |
|-----------|-----------|
| ≤ -0.5 | "Oh dear, that sounds bad" |
| ≤ 0 | "Hmm, that's not great" |
| ≤ 0.5 | "Well, that sounds positive" |
| ≤ 1 | "Wow, that sounds great" |

**¿Cómo calcula TextBlob la polaridad?** Usa un diccionario de palabras con pesos. "happy" = +0.8, "terrible" = -0.9, etc. Promedia todas las palabras.

### Paso 7: La pregunta inteligente

```python
if len(np) != 0:
    response = response + "Can you tell me more about " + np[0].pluralize() + "?"
else:
    response = response + "Can you tell me more?"
```

Si detectó frases sustantivas:
- Toma la primera (`np[0]`)
- La pluraliza (`"cat"` → `"cats"`, `"mouse"` → `"mice"`)
- Pregunta: "Can you tell me more about cats?"

Si no detectó nada → pregunta genérica.

### Paso 8: El goodbye

```python
print("It was nice talking to you, goodbye!")
```

Se imprime después de que el `while True` termina (cuando escribís "bye").

---

## ¿Por qué esto importa?

Este bot es la base de todo sistema de NLP:

1. **Entrada** → texto del usuario
2. **Procesamiento** → TextBlob analiza sentimiento + extrae entidades
3. **Decisión** → árbol de if/elif según la polaridad
4. **Salida** → respuesta generada

Es simplista, pero el mismo patrón se usa en:
- Chatbots de atención al cliente
- Análisis de redes sociales
- Sistemas de recomendación

La diferencia es que los bots reales usan **modelos de ML** (como GPT) en lugar de un diccionario de palabras.

---

## Errores comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'textblob'` | No instalado | `pip install textblob` |
| Bot no responde | `input()` en Jupyter muestra caja arriba | Mirá la parte superior del notebook |
| Sentimiento incorrecto | TextBlob no entiende sarcasmo | Es limitación del enfoque basado en diccionario |
| `np[0]` IndexError | No detectó noun phrases | El `if len(np) != 0` lo previene |

---

## Siguiente paso

En la **Lección 3** vamos a ver traducción automática y análisis de sentimiento en textos más largos — aplicado a *Orgullo y Prejuicio* de Jane Austen.
