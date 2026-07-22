# Word Embeddings — Incrustaciones de palabras

## ¿Qué son?

Los word embeddings son **representaciones numéricas de palabras**. Cada palabra se convierte en un vector (lista de números) de tal forma que **palabras con significado similar están cerca en el espacio vectorial**.

### Analogía simple

Imaginá un mapa donde cada palabra es una ciudad:
- "gato" y "gatito" están cerca 🐱
- "perro" y "cachorro" están cerca 🐕
- "gato" y "perro" están relativamente cerca (animales)
- "gato" y "automóvil" están lejos

```
        gato ●─────● gatito
              \
               \   
                \  
        perro ●───● cachorro
```

## ¿Por qué sirven?

Las computadoras **no entienden texto** — solo números. Los embeddings convierten palabras a números **preservando el significado**.

| Sin embedding | Con embedding |
|---------------|---------------|
| "gato" = 1 | "gato" = [0.2, 0.8, 0.1, ...] |
| "perro" = 2 | "perro" = [0.25, 0.75, 0.15, ...] |
| No hay relación | Los vectores son similares |

## Historia

| Año | Hito | Quién |
|-----|------|-------|
| 1957 | "Una palabra se conoce por la compañía que frecuenta" | John Rupert Firth |
| 2000 | Modelos de lenguaje probabilísticos neuronales | Yoshua Bengio |
| 2013 | **word2vec** — cambio todo | Tomas Mikolov (Google) |
| 2014 | **GloVe** — embeddings globales | Stanford |
| 2018 | **ELMo** — embeddings contextuales | AllenNLP |
| 2018 | **BERT** — revolución en NLP | Google |

## Herramientas populares

| Herramienta | Creador | Característica |
|-------------|---------|----------------|
| **word2vec** | Google | Rápido, popular |
| **GloVe** | Stanford | Basado en co-ocurrencia |
| **fastText** | Facebook | Maneja palabras raras |
| **BERT** | Google | Contextual, state-of-the-art |
| **Gensim** | Open source | Fácil de usar |

## Ejemplo práctico

```python
from gensim.models import Word2Vec

# Oraciones de entrenamiento
sentences = [
    ["gato", "gatito", "felino"],
    ["perro", "cachorro", "can"],
    ["manzana", "fruta", "comida"]
]

# Entrenar modelo
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Buscar palabras similares
similar = model.wv.most_similar("gato")
# Resultado: [("gatito", 0.9), ("felino", 0.85), ...]
```

## ¿Cómo se entrenan?

### Enfoque 1: Predictión de contexto (word2vec)

**Skip-gram**: Dada una palabra, predecir las palabras vecinas.

```
Oración: "el gato come pescado"
Ventana de 2: [el, gato, come, pescado]

Entrenamiento:
- Entrada: "gato"
- Salida esperada: "el", "come"
```

### Enfoque 2: Co-ocurrencia (GloVe)

Contar cuántas veces aparecen palabras juntas en un corpus grande.

```
           gato  perro  come  duerme
gato         10     2     8      3
perro         2    15     7      4
come          8     7     5      1
duerme        3     4     1     12
```

## Aplicaciones

| Aplicación | Cómo usa embeddings |
|------------|---------------------|
| **Análisis de sentimiento** | Palabras positivas cerca de "bueno", negativas cerca de "malo" |
| **Traducción** | Mapear vectores entre idiomas |
| **Búsqueda semántica** | Encontrar documentos similares sin palabras exactas |
| **Chatbots** | Entender el contexto de la conversación |
| **Clasificación de texto** | Agrupar documentos por tema |

## Limitaciones

### Polisemia (una palabra, varios significados)

"Club" puede significar:
- Sándwich club 🥪
- Casa club 🏠
- Palo de golf ⛳

Los embeddings estáticos (word2vec, GloVe) **fusionan todos los significados en un solo vector**.

### Solución: Embeddings contextuales (BERT)

BERT crea un **vector diferente para cada contexto**:
- "club" en "el club de golf" → vector A
- "club" en "el club nocturno" → vector B

## Sesgos éticos

Los embeddings aprenden **sesgos de los datos de entrenamiento**. Ejemplo famoso:

```
hombre es a programador como mujer es a ___
→ Resultado: "ama de casa"
```

Esto refleja sesgos sociales, no una verdad. Los investigadores trabajan en **debiasing** para eliminar estos sesgos.

## Resumen

| Concepto | Definición |
|----------|------------|
| **Word Embedding** | Representación numérica de palabras |
| **Vector** | Lista de números que codifica el significado |
| **Espacio vectorial** | Donde viven los embeddings (palabras similares = cerca) |
| **word2vec** | Primer embedding popular (2013, Google) |
| **BERT** | Embedding contextual (2018, Google) |
| **Sesgo** | Los embeddings pueden perpetuar estereotipos |

---

**Volver al [índice de NLP](README.md)**
