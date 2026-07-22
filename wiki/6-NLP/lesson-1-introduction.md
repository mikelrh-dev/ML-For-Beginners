# Lección 1: Introducción a NLP — Historia, Turing y Eliza

## ¿Qué aprendemos aquí?

De dónde viene el NLP, quiénes fueron los pioneros, y cómo funcionaba el primer "chatbot" de la historia — Eliza, un psicólogo fake que engañaba a la gente haciéndole creer que lo entendía.

---

## El problema fundamental

Las computadoras **no entienden lenguaje humano**. Solo entienden números. El NLP intenta cerrar esa brecha: ¿cómo hacer que una máquina "entienda" lo que le decís?

### Ejemplo cotidiano

Cuando le decís a Siri "Hey Siri, poné música", ella:
1. Convierte tu voz a texto
2. Analiza las palabras
3. Identifica la intención (reproducir música)
4. Ejecuta la acción

Todo eso es NLP en acción.

---

## Los pioneros

### Alan Turing (1950s)

**Pregunta**: ¿Puede una computadora "pensar"?

Turing propuso un test: si un humano conversa por escrito con algo (sin saber si es humano o máquina) y no puede distinguir la diferencia, entonces la máquina "piensa".

**El Juego de la Imitación**:
- Un interrogador está solo en una sala
- Puede hacer preguntas por escrito a dos personas en otra sala
- Debe adivinar cuál es hombre y cuál es mujer
- Las personas intentan confundirlo

Turing adaptó este juego para testear inteligencia artificial.

### Joseph Weizenbaum (1960s) — Eliza

Weizenbaum creó **Eliza** en MIT — un programa que simulaba ser un terapeuta.

**Cómo funcionaba Eliza:**

```
Humano: "I am sad"
Eliza:  "How long have you been sad?"

Humano: "My cat makes me happy"
Eliza:  "Tell me more about cats"
```

**El truco**: Eliza no "entendía" nada. Solo:
1. Buscaba palabras clave ("I am", "my")
2. Reorganizaba la frase cambiando tiempos verbales
3. Si no encontraba nada, daba una respuesta genérica

**Frase célebre**: Si le decías "You are a bicycle", respondía "How long have I been a bicycle?" — porque solo reemplazaba "You are" → "I been".

---

## El primer bot: código simple

El ejercicio de esta lección es crear un bot que responde con **frases aleatorias**:

```python
import random

random_responses = [
    "That is quite interesting, please tell me more.",
    "I see. Do go on.",
    "Why do you say that?",
    "Funny weather we've been having, isn't it?",
    "Let's change the subject.",
    "Did you catch the game last night?"
]

while True:
    user_input = input("> ")
    if user_input.lower() == "bye":
        break
    print(random(random_responses))
```

### ¿Por qué funciona (un poco)?

- Las frases son **genéricas** — sirven para cualquier contexto
- El humano **proyecta significado** — cree que el bot entiende
- Es el mismo principio que Eliza: la gente busca patrones donde no los hay

### Limitaciones obvias

| Lo que hace | Lo que no puede |
|-------------|-----------------|
| Responde algo | Entender el contenido |
| Mantiene conversación | Recordar contexto |
| Parece interactivo | Realmente procesar |

---

## Conceptos clave de la lección

| Concepto | Definición | Ejemplo |
|----------|------------|---------|
| **Computational Linguistics** | Cómo las computadoras trabajan con lenguaje | Google Translate, Siri |
| **Turing Test** | Test para determinar si una máquina "piensa" | Conversación escrita ciega |
| **Eliza** | Primer chatbot (1960s) | Simula terapeuta |
| **Parsing** | Analizar estructura gramatical | Identificar sustantivos, verbos |
| **Keyword matching** | Buscar palabras clave y responder | Eliza con "I am" → "How long..." |

---

## ¿Por qué importa esto?

1. **Historia**: Sin Eliza no existirían Siri, Alexa, ChatGPT
2. **Limitaciones**: El enfoque de "palabras clave" tiene techo — por eso necesitamos ML
3. **Ética**: Si un bot puede engañar gente haciéndole creer que es humano, ¿eso está bien?

---

## Errores comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'textblob'` | No instalado | `pip install textblob && python -m textblob.download_corpora` |
| Bot no entiende sarcasmo | Limitación del enfoque | Usar ML más avanzado (lecciones siguientes) |
| Confundir Eliza con IA real | Eliza es reglas, no ML | Eliza es un espejismo — reorganiza palabras, no entiende |

---

## Siguiente paso

En la **Lección 2** vemos cómo mejorar el bot: extraer frases sustantivas y analizar sentimiento real con TextBlob.
