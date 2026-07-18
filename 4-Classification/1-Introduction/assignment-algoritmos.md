# 5 Algoritmos de Clasificación — Guía de Estudio 📚

> Investigación para la lección 4-Classification/1-Introduction

---

## 1. Logistic Regression 🔢

**Aunque se llame "regresión", es un algoritmo de CLASIFICACIÓN.**

### ¿Cómo funciona?

Dibuja una **frontera de decisión** (una línea o hiperplano) que mejor separa las clases. Para cada punto, calcula una **probabilidad** de pertenecer a cada clase:
- Si probabilidad > 0.5 → asigna una clase
- Si probabilidad ≤ 0.5 → asigna la otra

Es como un clasificador "suave": está seguro lejos de la frontera y dudoso cerca de ella.

### ¿Cuándo usarlo?

- Clasificación binaria (sí/no, spam/no-spam)
- Cuando necesitás entender qué features impulsan la decisión
- Como **baseline rápido** antes de probar modelos más complejos

### Ventajas
- ✅ Rápido de entrenar y predecir
- ✅ Altamente interpretable (los coeficientes tienen significado)
- ✅ Funciona bien cuando las clases son linealmente separables

### Desventajas
- ❌ Asume frontera lineal — falla en relaciones no lineales
- ❌ Puede fallar con muchas features correlacionadas
- ❌ Necesita suficientes muestras por feature

### Import en sklearn
```python
from sklearn.linear_model import LogisticRegression
```

### Dataset del curso: OVNIs 🛸

Usando `ufos.csv`, predecir si un avistamiento ocurrió en **EE.UU.** o no, basándose en duración, latitud y longitud.

**Pregunta:** *"Dada la duración y la ubicación geográfica, ¿es este avistamiento de EE.UU. o de otro país?"*

**Explicación:** Usaría Logistic Regression porque la frontera entre EE.UU. y otros países es aproximadamente una línea geográfica. El modelo aprendería que ciertas coordenadas y duraciones son más probables de ser reportes estadounidenses.

---

## 2. K-Nearest Neighbors (KNN) 👥

### ¿Cómo funciona?

**Memoriza** todo el dataset de entrenamiento. Cuando llega un punto nuevo:
1. Busca los `k` puntos más cercanos (por distancia euclidiana)
2. Cuenta votos: la clase ganadora entre los vecinos gana

No "aprende" nada — es un algoritmo **perezoso** que solo almacena datos.

### ¿Cuándo usarlo?

- Datasets pequeños/medianos
- Fronteras de decisión irregulares o no lineales
- No conocés la distribución de los datos

### Ventajas
- ✅ Sin fase de entrenamiento — solo almacena
- ✅ Maneja problemas multiclase naturalmente
- ✅ Funciona para fronteras no lineales sin transformaciones

### Desventajas
- ❌ Predicción lenta (calcula distancia a cada punto)
- ❌ sensible a la escala de features — requiere normalización
- ❌ Falla con dimensiones altas (curse of dimensionality)

### Import en sklearn
```python
from sklearn.neighbors import KNeighborsClassifier
```

### Dataset del curso: Cocinas Asiáticas 🍜

Usando `cuisines.csv`, predecir el tipo de cocina basándose en los 384 ingredientes.

**Pregunta:** *"Dados estos ingredientes, ¿de qué cocina es esta receta?"*

**Explicación:** Usaría KNN porque tiene sentido intuitivo: si una receta tiene ingredientes similares a recetas indias conocidas, probablemente sea india. KNN mide la "distancia" entre recetas y asigna la cocina más cercana.

---

## 3. Support Vector Machines (SVM) 🎯

### ¿Cómo funciona?

Busca el **hiperplano** que separe las clases con el **mayor margen** posible. Se enfoca en los puntos más difíciles de clasificar — los **support vectors** que están cerca de la frontera.

Con el **kernel trick**, puede proyectar datos a dimensiones superiores para encontrar separaciones que no son lineales en el espacio original.

### ¿Cuándo usarlo?

- Datos de **alta dimensionalidad** (más features que muestras)
- Existe un **margen claro** de separación
- Clasificación de texto, imágenes

### Ventajas
- ✅ Funciona bien en espacios de alta dimensión
- ✅ Eficiente en memoria — usa solo un subconjunto de puntos
- ✅ El kernel trick maneja fronteras no lineales

### Desventajas
- ❌ No produce probabilidades directamente
- ❌ Sensible a la escala de features
- ❌ Selección de kernel e hiperparámetros es compleja
- ❌ Lento de entrenar en datasets grandes

### Import en sklearn
```python
from sklearn.svm import SVC
```

### Dataset del curso: Cocinas Asiáticas 🍜

Usando `cuisines.csv`, clasificar cocinas basándose en ingredientes (alta dimensión, datos dispersos).

**Pregunta:** *"Dado este vector de 384 ingredientes, ¿qué cocina es?"*

**Explicación:** SVM es mejor que KNN aquí porque los datos son de alta dimensión (384 ingredientes) y dispersos (la mayoría son 0). El kernel RBF maneja la relación no lineal entre ingredientes y cocinas (ej: el coco puede ser indio o tailandés dependiendo con qué se combine).

---

## 4. Decision Trees 🌳

### ¿Cómo funciona?

Construye un **árbol de preguntas sí/no**. En cada nodo, picka la feature y umbral que mejor divide los datos en grupos puros.

- Arriba: pregunta más importante
- Abajo: preguntas más finas

Para clasificar, caminás de la raíz hasta una hoja siguiendo las preguntas.

### ¿Cuándo usarlo?

- Necesitás un modelo **explicable** a personas no técnicas
- El proceso de decisión debe ser auditable
- Features de diferentes escalas o tipos mixtos

### Ventajas
- ✅ No necesita normalización de features
- ✅ Captura relaciones no lineales naturalmente
- ✅ El árbol resultante es interpretable — podés trazar cualquier predicción

### Desventajas
- ❌ Muy propenso a overfitting (árboles profundos memorizan ruido)
- ❌ Cambios pequeños en datos → árboles muy diferentes
- ❌ Puede crear árboles sesgados si algunas clases dominan

### Import en sklearn
```python
from sklearn.tree import DecisionTreeClassifier
```

### Dataset del curso: Calabazas 🎃

Usando `US-pumpkins.csv`, predecir si una calabaza es **cara o barata** (binning el precio) basándose en ciudad, color, tamaño, variedad, origen.

**Pregunta:** *"Dada la ciudad, el color y el tamaño, ¿es esta calabaza cara o barata?"*

**Explicación:** Usaría Decision Tree porque maneja mezcla de categóricas (ciudad, color, origen) y numéricas (tamaño) sin necesitar escalado. El árbol podría revelar reglas como: "si origen=MARYLAND y color=ORANGE y size=lge → cara". Perfecto para un modelo explicable.

---

## 5. Random Forest 🌲🌲🌲

### ¿Cómo funciona?

Entrena **cientos de árboles de decisión**, cada uno en:
- Una **muestra aleatoria** de los datos (bootstrap)
- Un **subconjunto aleatorio** de features en cada split

Cada árbol vota, y la **mayoría gana**. El "sabiduría de la multitud" cancela el overfitting individual de cada árbol.

### ¿Cuándo usarlo?

- **Default go-to** para datos tabulares
- Querés alta precisión sin mucho tuning de hiperparámetros
- Importa la **importancia de features** (saber qué es predictivo)

### Ventajas
- ✅ Robusto contra overfitting (a diferencia de un solo árbol)
- ✅ Maneja features numéricas y categóricas
- ✅ Da importancia de features de gratis
- ✅ Funciona bien con defaults

### Desventajas
- ❌ No interpretable (cientos de árboles = caja negra)
- ❌ Modelo más grande y más lento de predecir que un solo árbol
- ❌ Puede overfitting en datos ruidosos si no se limita `max_depth`

### Import en sklearn
```python
from sklearn.ensemble import RandomForestClassifier
```

### Dataset del curso: Calabazas 🎃

Usando `US-pumpkins.csv`, predecir **calidad o tamaño** basándose en color, variedad, origen, tamaño y precio.

**Pregunta:** *"Dadas estas características, ¿cuál es la calidad esperada de esta calabaza?"*

**Explicación:** Random Forest sobresale aquí porque modela interacciones complejas (ej: una calabaza naranja de Maryland podría ser de alta calidad, pero una de Texas no). También da importancia de features para ver qué factor más influye en la calidad. Ideal como enfoque de "lanzá los datos y veamos qué pasa".

---

## Tabla Resumen

| Algoritmo | Import | Mejor para | Sklearn Class |
|-----------|--------|------------|---------------|
| Logistic Regression | `sklearn.linear_model` | Binario, baselines interpretables | `LogisticRegression` |
| KNN | `sklearn.neighbors` | Datasets pequeños, no lineal, multiclase | `KNeighborsClassifier` |
| SVM | `sklearn.svm` | Alta dimensión, márgenes claros | `SVC` |
| Decision Tree | `sklearn.tree` | Explicabilidad, tipos mixtos | `DecisionTreeClassifier` |
| Random Forest | `sklearn.ensemble` | Datos tabulares, default go-to, importancia | `RandomForestClassifier` |

## Matriz Dataset-Algoritmo

| Dataset | Filas | Features | Target | Mejor match | Por qué |
|---------|-------|----------|--------|-------------|---------|
| **OVNIs** | ~80k | lat, lon, seconds, country | country (US/no) | Logistic Regression | Frontera lineal, binario, rápido |
| **Cocinas** | ~250 | 384 ingredientes | tipo de cocina (5 clases) | SVM / Random Forest | Alta dimensión disperso; RF para importancia |
| **Calabazas** | ~1800 | Ciudad, Color, Size, Precio | Calidad / Precio | Random Forest / Decision Tree | Tipos mixtos, interacciones, interpretabilidad |

---

## Flujo de Decisión: ¿Qué algoritmo elijo?

```
¿Es binario?
├── Sí → ¿Necesitás interpretabilidad?
│   ├── Sí → Logistic Regression
│   └── No → SVM o Random Forest
└── No (multiclase)
    ├── Dataset pequeño (<5k filas)?
    │   ├── Sí → KNN
    │   └── No → Random Forest
    ├── ¿Necesitás explicar el modelo?
    │   ├── Sí → Decision Tree
    │   └── No → Random Forest
    └── Alta dimensionalidad?
        ├── Sí → SVM
        └── No → Random Forest
```
