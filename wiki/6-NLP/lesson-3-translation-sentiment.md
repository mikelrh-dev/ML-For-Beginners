# Lección 3: Traducción y Análisis de Sentimiento

## ¿Qué aprendemos aquí?

Dos capacidades fundamentales del NLP:
1. **Traducción automática** — convertir texto de un idioma a otro
2. **Análisis de sentimiento** — determinar si un texto es positivo o negativo

Ambas aplicadas a *Orgullo y Prejuicio* de Jane Austen.

---

## El problema de la traducción

### No es solo reemplazar palabras

Cada idioma tiene gramática diferente. Ejemplo:

| Inglés | Irlandés | Literal |
|--------|----------|---------|
| I feel happy | Tá athas orm | Happy is upon me |

En irlandés, las emociones **están sobre vos**, no las "sentís". Un traductor literal falla.

### Enfoque tradicional vs ML

| Enfoque | Cómo funciona | Limitación |
|---------|---------------|------------|
| **Reglas** | Identificar palabras → traducir una por una | Ignora gramática |
| **ML** | Entrenar con millones de traducciones humanas | Necesita datos masivos |

Google Translate usa **ML** entrenado con millones de traducciones humanas.

### Ejemplo práctico

```python
from deep_translator import GoogleTranslator

traductor = GoogleTranslator(source='en', target='fr')

original = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
traduccion = traductor.translate(original)

print(traduccion)
# C'est une vérité universellement reconnue, qu'un homme célibataire...
```

**¿Por qué `deep-translator` y no `TextBlob`?** TextBlob eliminó su función `translate()` en versiones recientes (0.20.x) porque Google bloqueó el acceso no oficial. `deep-translator` usa la API de Google de forma legal.

**Dato curioso**: La traducción automática es más precisa que la traducción humana de 1932, que agregaba palabras innecesarias.

---

## Análisis de Sentimiento

### ¿Qué es?

Determinar si un texto es **positivo**, **negativo** o **neutral** usando números.

### Métricas

| Métrica | Rango | Significado |
|---------|-------|-------------|
| **Polaridad** | -1 a +1 | -1 = muy negativo, +1 = muy positivo |
| **Subjetividad** | 0 a 1 | 0 = hecho objetivo, 1 = opinión subjetiva |

### El diccionario de pesos

TextBlob tiene un diccionario donde cada palabra tiene un peso:

```
"happy"     = +0.8
"terrible"  = -1.0
"amazing"   = +0.6
"hate"      = -0.8
"fine"      = +0.42
"wonderful" = +1.0
```

**Cálculo**: Promedio de los pesos de las palabras con significado.

```
"I love this amazing day"
  "love" = +0.50
  "amazing" = +0.60
  promedio = (0.50 + 0.60) / 2 = 0.55 → "Well, that sounds positive"
```

### Ejemplo con Orgullo y Prejuicio

```python
from textblob import TextBlob

quote1 = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
quote2 = "Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."

# Quote 1: polaridad 0.21, subjetividad 0.27
# → Observación general, no personal

# Quote 2: polaridad 0.70, subjetividad 0.80
# → Habla de amor y gratitud, claramente emocional
```

---

## El problema del sarcasmo

### Frases "trampa" de Jane Austen

| Frase | Polaridad | ¿Correcto? |
|-------|-----------|------------|
| "Happy shall I be, when his stay at Netherfield is over!" | +1.0 | ❌ Es sarcasmo |
| "Our distress, my dear Lizzy, is very great." | +0.5 | ❌ Es angustia |
| "I have the greatest dislike in the world to that sort of thing." | +1.0 | ❌ Es negativo |

### ¿Por qué falla?

1. **No entiende contexto**: "Happy" aparece, pero la estructura implica impaciencia
2. **No detecta sarcasmo**: "distress" es negativo, pero "my dear" confunde
3. **Palabras individuales engañan**: El análisis palabra por palabra falla con construcciones complejas

---

## Pipeline completo de la lección

```
Texto original (Pride & Prejudice)
    ↓ TextBlob
Análisis de sentimiento por oración
    ↓ Clasificación
Frases positivas (polaridad = 1)
Frases negativas (polaridad = -1)
    ↓ Resultado
Estadísticas: ¿más positivas o negativas?
```

---

## Aplicaciones reales

| Aplicación | Cómo usa sentimiento |
|------------|---------------------|
| **Análisis de redes sociales** | Medir percepción de marca |
| **Atención al cliente** | Clasificar emails urgentes |
| **Política** | Leer emails de ciudadanos a favor/en contra |
| **Marketing** | Medir reacción a campañas |

---

## Conceptos clave

| Concepto | Definición |
|----------|------------|
| **Polaridad** | Qué tan positivo/negativo es un texto (-1 a +1) |
| **Subjetividad** | Qué tan objetivo/subjetivo es (0 a 1) |
| **Corpus** | Colección de textos para entrenar |
| **False cognate** | Palabras que parecen iguales en dos idiomas pero significan diferente |
| **Sarcasmo** | Ironía que engaña al análisis palabra por palabra |

---

## Errores comunes

| Error | Causa | Solución |
|-------|-------|----------|
| Sentimiento incorrecto | TextBlob no entiende sarcasmo | Usar VADER o modelos contextuales |
| `ModuleNotFoundError: No module named 'deep_translator'` | No instalado | `pip install deep-translator` |
| `blob.translate()` no existe | TextBlob 0.20+ quitó translate() | Usar `deep-translator` en su lugar |
| Polaridad = 0 en todo | Texto sin palabras del diccionario | Es limitación del enfoque |

---

## Siguiente paso

En la **Lección 4** empezamos a trabajar con un dataset real de 515K reseñas de hoteles — exploración de datos antes de aplicar NLP.
