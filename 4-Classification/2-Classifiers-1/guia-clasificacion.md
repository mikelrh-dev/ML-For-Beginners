# Guía Completa de Clasificación 🎓

> Todo lo que necesitás saber para entender clasificación ML desde cero.
> Subí este archivo a NotebookLM para generar presentaciones, quizzes y más.

---

## 1. ¿Qué es la Clasificación?

### La idea central

La clasificación es una de las tareas fundamentales del Machine Learning. Consiste en **predecir a qué categoría o clase pertenece un dato**, basándose en ejemplos anteriores.

### Analogía cotidiana

Imaginás que sos un chef experimentado. Cuando ves una receta con ciertos ingredientes, podés decir "esto es tailandés" o "esto es indio" porque ya comiste y cocinaste muchas veces. Un modelo de clasificación hace exactamente lo mismo: aprende de ejemplos para poder clasificar cosas nuevas.

### Clasificación binaria vs multiclase

| Tipo | Ejemplo | Salida |
|------|---------|--------|
| **Binaria** | ¿Este email es spam? | Sí / No |
| **Multiclase** | ¿De qué cocina es esta receta? | Tailandés / Japonés / Chino / Indio / Coreano |

En este curso trabajamos con **clasificación multiclase** — predecir entre 5 cocinas posibles.

---

## 2. El Flujo Completo de un Modelo de Clasificación

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│ Datos crudos │ →  │ Limpiar y    │ →  │ Dividir     │ →  │ Entrenar │ →  │ Evaluar  │
│ (CSV)        │    │ preparar     │    │ train/test  │    │ modelo   │    │ modelo   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘    └──────────┘
```

### Paso 1: Datos cruros
Tenés un archivo CSV con filas (ejemplos) y columnas (features + label).

### Paso 2: Limpiar
- Quitar filas con datos faltantes
- Quitar columnas que no aportan (IDs, etc.)
- Balancear clases (SMOTE si hay desbalance)
- Codificar categorías a números (LabelEncoder)

### Paso 3: Dividir
Separás los datos en:
- **Train (70-80%)**: Para entrenar el modelo
- **Test (20-30%)**: Para evaluar el modelo

¿Por qué? Porque si evaluás con los mismos datos de entrenamiento, el modelo "hace trampa" — memoriza las respuestas en vez de aprender.

### Paso 4: Entrenar
El modelo encuentra patrones en los datos de entrenamiento. Cada algoritmo lo hace de forma diferente (esto es lo que los distingue).

### Paso 5: Evaluar
Medís qué tan bien funciona con datos que **nunca vio**. Si el accuracy es alto, el modelo aprendió bien. Si es bajo, algo falló.

---

## 3. Logistic Regression — Explicación Detallada

### ¿Qué es?

A pesar de llamarse "regresión logística", es un algoritmo de **clasificación**. Es el punto de partida más común en ML porque es simple, rápido e interpretable.

### ¿Cómo funciona? (Sin fórmulas)

Imaginás que tenés datos de dos cocinas: tailandesa e india. Cada receta tiene ingredientes (features). Logistic Regression dibuja una **frontera** que mejor separa las dos cocinas.

Cuando llega una receta nueva:
1. Mira de qué lado de la frontera cae
2. Calcula qué tan lejos está de la frontera
3. Convierte esa distancia en una **probabilidad** (0 a 1)
4. Si la probabilidad es > 0.5, predice una clase; si no, la otra

### ¿Qué es la frontera?

Es la línea que separa las clases. En 2D es literalmente una línea. En más dimensiones es un hiperplano.

### ¿Qué es Sigmoid?

Es una función que convierte cualquier número en una probabilidad (entre 0 y 1):

```
σ(x) = 1 / (1 + e^(-x))
```

No necesitás memorizar la fórmula. Solo sabé que:
- Si el número es muy negativo → probabilidad cercana a 0
- Si el número es muy positivo → probabilidad cercana a 1
- Si es 0 → probabilidad exactamente 0.5

### Multiclase: One-vs-Rest (OvR)

Logistic Regression originalmente es binaria (2 clases). Para manejar 5 cocinas, sklearn usa **One-vs-Rest**:

1. Entrena un clasificador: "tailandés vs el resto"
2. Entrena otro: "japonés vs el resto"
3. Entrena otro: "chino vs el resto"
4. Entrena otro: "indio vs el resto"
5. Entrena otro: "coreano vs el resto"

Cuando predice, cada clasificador vota y gana el que tiene mayor probabilidad.

### ¿Qué es un Solver?

El solver es el **algoritmo matemático** que encuentra los mejores coeficientes para la frontera de decisión. No es el modelo — es el método para optimizarlo.

| Solver | Tipo | Mejor para | Desventaja |
|--------|------|------------|------------|
| `liblinear` | Coordenada descendente | Datasets pequeños (<10k filas) | No maneja multinomial |
| `lbfgs` | Quasi-Newton | Datasets medianos, multiclass | Puede no converger en datos muy grandes |
| `saga` | SGD | Datasets grandes, sparse | Necesita más iteraciones |
| `newton-cg` | Newton | Precisión alta | Lento en muchos features |

**Regla simple**: Si tu dataset tiene menos de 10,000 filas, usá `liblinear`. Si es más grande, usá `lbfgs`.

---

## 4. Métricas de Evaluación — Explicación Detallada

### Accuracy (Precisión general)

```
Accuracy = (predicciones correctas) / (total de predicciones)
```

Es la métrica más simple. Si tu modelo acierta 80 de 100 veces, el accuracy es 80%.

**Problema**: Si el dataset está desbalanceado (ej: 90% de una clase), un modelo tonto que siempre prediga esa clase tendría 90% de accuracy sin aprender nada.

### Precision (Precisión por clase)

```
Precision = (correctos de esta clase) / (todos los predichos como esta clase)
```

Pregunta: "De todos los que DIJE que eran tailandeses, ¿cuántos realmente lo eran?"

**Cuándo importa**: Cuando el costo de un falso positivo es alto. Ej: si diagnosticás una enfermedad que no existe, el paciente sufre innecesariamente.

### Recall (Sensibilidad)

```
Recall = (correctos de esta clase) / (todos los que realmente son esta clase)
```

Pregunta: "De todos los que REALMENTE son tailandeses, ¿cuántos detecté?"

**Cuándo importa**: Cuando el costo de un falso negativo es alto. Ej: si no detectás una enfermedad, el paciente no recibe tratamiento.

### F1-Score (Balance)

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Es el promedio armónico entre precision y recall. Si ambos son altos, F1 es alto. Si uno es bajo, F1 baja.

**Cuándo usarlo**: Cuando necesitás balance entre precision y recall, y no sabés cuál es más importante.

### Confusion Matrix (Matriz de Confusión)

Es una tabla que muestra dónde se equivoca el modelo:

```
              Predicho
              Thai  Jap  Chi  Ind  Cor
Real Thai  [  45    3    2    0    0  ]
Real Jap   [   2   40    5    1    2  ]
Real Chi   [   1    4   38    3    4  ]
Real Ind   [   0    1    2   46    1  ]
Real Cor   [   1    2    3    1   43  ]
```

- **Diagonal principal** (45, 40, 38, 46, 43): Predicciones correctas
- **Fuera de diagonal**: Errores de clasificación
- **Filas**: Datos reales
- **Columnas**: Predicciones del modelo

Si ves muchos errores entre chino y japonés, es porque comparten muchos ingredientes.

---

## 5. Overfitting y Underfitting

### Overfitting (Sobreajuste)

El modelo **memoriza** los datos de entrenamiento en vez de aprender patrones. Como un estudiante que memoriza respuestas en vez de entender.

**Síntomas**:
- Accuracy alto en train, bajo en test
- El modelo es demasiado complejo

**Soluciones**:
- Más datos de entrenamiento
- Simplificar el modelo
- Regularización (parámetros C en LogisticRegression)

### Underfitting (Subajuste)

El modelo es **demasiado simple** para capturar los patrones. Como intentar explicar un tema complejo con una frase.

**Síntomas**:
- Accuracy bajo en train Y en test
- El modelo no aprende nada

**Soluciones**:
- Modelo más complejo
- Más features
- Menos regularización

### El punto dulce

El objetivo es encontrar el equilibrio: un modelo que sea lo suficientemente complejo para aprender patrones, pero no tanto como para memorizar ruido.

---

## 6. Comparación de Algoritmos

### Logistic Regression

| Aspecto | Detalle |
|---------|---------|
| **Qué hace** | Dibuja una frontera lineal |
| **Ventaja** | Rápido, interpretable |
| **Desventaja** | Solo fronteras lineales |
| **Cuándo usar** | Baseline, datos lineales |
| **sklearn** | `LogisticRegression()` |

### K-Nearest Neighbors (KNN)

| Aspecto | Detalle |
|---------|---------|
| **Qué hace** | Busca los k vecinos más cercanos y vota |
| **Ventaja** | No necesita entrenamiento |
| **Desventaja** | Lento en predicción, sensible a escala |
| **Cuándo usar** | Datasets pequeños, fronteras irregulares |
| **sklearn** | `KNeighborsClassifier()` |

### Support Vector Machines (SVM)

| Aspecto | Detalle |
|---------|---------|
| **Qué hace** | Encuentra el hiperplano con mayor margen |
| **Ventaja** | Funciona en alta dimensión |
| **Desventaja** | Lento de entrenar, difícil de interpretar |
| **Cuándo usar** | Alta dimensionalidad, márgenes claros |
| **sklearn** | `SVC()` |

### Decision Tree

| Aspecto | Detalle |
|---------|---------|
| **Qué hace** | Construye un árbol de preguntas sí/no |
| **Ventaja** | Totalmente interpretable |
| **Desventaja** | Muy propenso a overfitting |
| **Cuándo usar** | Necesitás explicar el modelo |
| **sklearn** | `DecisionTreeClassifier()` |

### Random Forest

| Aspecto | Detalle |
|---------|---------|
| **Qué hace** | Entrena muchos árboles y vota la mayoría |
| **Ventaja** | Robusto, maneja todo tipo de datos |
| **Desventaja** | Caja negra, lento en predicción |
| **Cuándo usar** | Default para datos tabulares |
| **sklearn** | `RandomForestClassifier()` |

---

## 7. Árbol de Decisión: ¿Qué Algoritmo Elegir?

```
¿Tu problema es?
├── Binario (sí/no)
│   ├── ¿Necesitás interpretabilidad?
│   │   ├── Sí → Logistic Regression
│   │   └── No → SVM o Random Forest
│   └── ¿Tenés muchos datos (>10k)?
│       ├── Sí → Random Forest
│       └── No → Logistic Regression
│
└── Multiclase (más de 2 opciones)
    ├── ¿Dataset pequeño (<5k)?
    │   ├── Sí → KNN
    │   └── No → Random Forest
    ├── ¿Necesitás explicar el modelo?
    │   ├── Sí → Decision Tree
    │   └── No → Random Forest
    └── Alta dimensionalidad (>100 features)?
        ├── Sí → SVM
        └── No → Random Forest
```

**Regla de oro**: Si no sabés qué elegir, empezá con Logistic Regression como baseline. Si no funciona bien, probá Random Forest.

---

## 8. Errores Comunes y Soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `ConvergenceWarning` | Modelo no convergió | Aumentar `max_iter=1000` |
| Accuracy muy bajo (<60%) | Modelo no aprende | Más features, otro algoritmo, más datos |
| Accuracy muy alto (>99%) | Overfitting o data leakage | Verificar train/test split |
| `ValueError: could not convert` | Strings sin codificar | `LabelEncoder` o `OneHotEncoder` |
| Accuracy desbalanceado por clase | Datos desbalanceados | SMOTE, `class_weight='balanced'` |
| Todas las filas predicen lo mismo | Features no informativas | Revisar qué columnas aportan |

---

## 9. Glosario Rápido

| Término | Definición en una línea |
|---------|------------------------|
| **Feature** | Variable de entrada (columna) |
| **Label** | Variable a predecir (target) |
| **Training** | Entrenar el modelo con datos conocidos |
| **Testing** | Evaluar el modelo con datos desconocidos |
| **Accuracy** | % de predicciones correctas |
| **Precision** | De los predichos como X, ¿cuántos son X? |
| **Recall** | De los que son X, ¿cuántos detectó? |
| **F1-score** | Balance entre precision y recall |
| **Confusion Matrix** | Tabla de aciertos y errores por clase |
| **Solver** | Algoritmo de optimización del modelo |
| **Overfitting** | Modelo memoriza, no aprende |
| **Underfitting** | Modelo demasiado simple |
| **SMOTE** | Técnica para balancear datos |
| **OvR** | One-vs-Rest: estrategia para multiclase |
| **Baseline** | Modelo simple para comparar contra |

---

## 10. Prompts para NotebookLM

Usá estos prompts en NotebookLM para generar materiales de estudio:

### Presentación general
```
Genera una presentación que explique clasificación ML desde cero.
Empezá con qué es la clasificación, después explique el flujo completo
(datos → limpiar → dividir → entrenar → evaluar). Usá analogías
cotidianas. Incluí una diapositiva por cada métrica (accuracy,
precision, recall, F1) con un ejemplo práctico.
```

### Guía de métricas
```
Creá una presentación que explique SOLO las métricas de evaluación:
accuracy, precision, recall, F1-score y confusion matrix.
Para cada una: qué mide, cuándo usarla, y un ejemplo numérico
ficticio donde se vea la diferencia entre ellas.
```

### Comparativa de algoritmos
```
Generá una comparación visual entre Logistic Regression, KNN, SVM,
Decision Tree y Random Forest. Para cada uno: cómo funciona
(explicación intuitiva), ventajas, desventajas, y cuándo usarlo.
Incluí un árbol de decisión final: ¿qué algoritmo elijo según mi problema?
```

### Quiz interactivo
```
Creá un quiz de 10 preguntas sobre clasificación ML:
- 3 sobre qué es la clasificación
- 3 sobre métricas
- 2 sobre Logistic Regression
- 2 sobre elegir algoritmos
Cada pregunta con 4 opciones y la respuesta correcta explicada.
```

### Cheat sheet
```
Generá un cheat sheet de una página con:
- Flujo completo de clasificación (5 pasos)
- Métricas: definición rápida
- Algoritmos: cuándo usar cada uno
- Errores comunes y soluciones
- Funciones de sklearn con su import
```
