# Introducción al procesamiento de lenguaje natural

Esta lección cubre una breve historia y conceptos importantes del *procesamiento de lenguaje natural* (NLP), un subcampo de la *lingüística computacional*.

## [Quiz pre-lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

NLP, como se le conoce comúnmente, es una de las áreas más conocidas donde el machine learning ha sido aplicado y utilizado en software de producción.

¿Usás software todos los días que probablemente tiene NLP? Tu procesador de textos o apps del celular seguramente lo tienen.

Vas a aprender sobre:

- **La idea de los idiomas**. Cómo se desarrollaron los idiomas y cuáles han sido las principales áreas de estudio.
- **Definiciones y conceptos**. Cómo las computadoras procesan texto, incluyendo parsing, gramática, y identificación de sustantivos y verbos.

## Lingüística computacional

La lingüística computacional es un área de investigación de varias décadas que estudia cómo las computadoras pueden trabajar con idiomas: comprenderlos, traducirlos y comunicarse en ellos.

### Ejemplo: dictado por teléfono

Si alguna vez le dictaste a tu teléfono en vez de escribir, o le preguntaste algo a un asistente virtual, tu voz se convirtió en texto y luego fue procesada o *parseada*.

### ¿Cómo es posible esta tecnología?

Alguien escribió un programa para hacer esto. Hace décadas, algunos escritores de ciencia ficción predijeron que hablaríamos con nuestras computadoras y siempre nos entenderían. Resultó ser más difícil de lo imaginado, pero hoy es mucho mejor.

No te preocupes si no sos experto en diferenciar sustantivos de verbos — las computadoras son buenas aplicando reglas formales, y vas a aprender a escribir código que pueda *parsear* oraciones.

### Requisitos previos

- **Python 3**: input, loops, lectura de archivos, arrays
- **TextBlob**: librería de procesamiento de texto

```bash
pip install -U textblob
python -m textblob.download_corpora
```

## Hablando con máquinas

La historia de intentar hacer que las computadoras entiendan el lenguaje humano tiene décadas, y uno de los primeros científicos fue *Alan Turing*.

### El test de Turing

En los años 50, Turing investigaba *inteligencia artificial* y consideró si una prueba conversacional podría darse a un humano y una computadora donde el humano no estuviera seguro con quién está hablando.

Si después de cierto tiempo el humano no pudiera determinar si las respuestas eran de una computadora, ¿podría decirse que la computadora está *pensando*?

### El juego de la imitación

La idea vino de un juego de fiesta llamado *El juego de la imitación* donde un interrogador debe determinar quién es hombre y quién es mujer entre dos personas en otra habitación.

### Eliza

En los años 60, *Joseph Weizenbaum* del MIT desarrolló [*Eliza*](https://wikipedia.org/wiki/ELIZA), una computadora "terapeuta" que hacía preguntas y daba la apariencia de entender las respuestas.

Si Eliza recibía "Yo estoy triste", respondía "¿Cuánto tiempo llevas triste?" — cambiaba el tiempo y agregaba palabras. No *entendía*, solo reorganizaba texto.

[![Charlando con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Charlando con Eliza")

> 🎥 Click en la imagen para ver un video sobre el programa ELIZA original

## Ejercicio: crear un bot conversacional

Un bot conversacional es un programa que recibe entrada del usuario y parece entender y responder inteligentemente.

### El plan

1. Mostrar instrucciones al usuario
2. Empezar un loop
   1. Aceptar entrada del usuario
   2. Si el usuario quiere salir, salir
   3. Procesar la entrada y determinar respuesta
   4. Mostrar respuesta
3. Repetir el paso 2

### Construyendo el bot

```python
random_responses = ["That is quite interesting, please tell me more.",
                    "I see. Do go on.",
                    "Why do you say that?",
                    "Funny weather we've been having, isn't it?",
                    "Let's change the subject.",
                    "Did you catch the game last night?"]
```

Ejemplo de salida:
```
Hello, I am Marvin, the simple robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am good thanks
That is quite interesting, please tell me more.
> today I went for a walk     
Did you catch the game last night?
> bye
It was nice talking to you, goodbye!
```

Una solución posible está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

### Reflexión

1. ¿Las respuestas aleatorias "engañarían" a alguien para que piense que el bot lo entiende?
2. ¿Qué características necesitaría el bot para ser más efectivo?
3. Si un bot pudiera "entender" el significado de una oración, ¿necesitaría "recordar" oraciones anteriores?

---

## 🚀Desafío

Elegí uno de los elementos de "reflexión" e intenta implementarlo en código.

En la siguiente lección, aprenderás sobre otras técnicas para parsear lenguaje natural.

## [Quiz post-lección](https://ff-quizzes.netlify.app/en/ml/)

## Tarea

[Buscar un bot](assignment.md)
