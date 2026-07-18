# 3. Comparando algoritmos 🏆

> "No hay un algoritmo mejor universal — solo el mejor para TUS datos."

## ¿Qué aprendemos aquí?

Probamos 5 algoritmos diferentes y comparamos cuál funciona mejor.
La lección: siempre probá varios antes de decidir.

## El código

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

classifiers = {
    'Linear SVC': SVC(kernel='linear', C=10, probability=True),
    'KNN': KNeighborsClassifier(10),
    'SVC': SVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=100)
}

for name, clf in classifiers.items():
    clf.fit(X_train, np.ravel(y_train))
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}: {accuracy:.1%}')
```

## Resultados esperados

| Algoritmo | Accuracy | ¿Por qué? |
|-----------|----------|------------|
| Linear SVC | ~78% | Frontera lineal — no captura complejidad |
| KNN | ~74% | Curse of dimensionality (384 features) |
| SVC (RBF) | ~83% | Kernel RBF captura fronteras no lineales |
| **Random Forest** | **~84%** | **Cientos de árboles votan** |
| AdaBoost | ~72% | Propenso a overfitting |

## ¿Por qué gana Random Forest?

1. **Ensemble**: Cientos de árboles deciden juntos
2. **Interacciones**: Captura combinaciones de ingredientes
3. **Robustez**: Un solo árbol se equivoca, pero 100 no

## Parámetros importantes

| Algoritmo | Parámetro | Qué controla |
|-----------|-----------|--------------|
| Random Forest | `n_estimators` | Número de árboles |
| SVC | `kernel` | Función de mapeo |
| KNN | `n_neighbors` | Cuántos vecinos mira |

## Conceptos clave

- **Ensemble**: Combinar múltiples modelos
- **Kernel**: Función que mapea datos a dimensiones superiores
- **Curse of dimensionality**: Con muchas features, las distancias pierden sentido
- **Regularization (C)**: Controla la complejidad del modelo

## Flujo de decisión

```
¿Qué problema tenés?
├── Binario, interpretabilidad → Logistic Regression
├── Datos pequeños, no lineal → KNN
├── Alta dimensión → SVC
├── Datos tabulares, default → Random Forest
└── No sabés → Probá todos y compará
```
