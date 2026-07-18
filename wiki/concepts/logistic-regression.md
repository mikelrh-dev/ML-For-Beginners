# Logistic Regression 🏷️

> "Clasificar es poner cosas en cajas."

## ¿Qué es?

Un modelo que predice **categorías**, no números.
Aprende fronteras entre clases.

## Analogía

Imagina que tienes manzanas y naranjas.
El modelo aprende a distinguirlas por color, tamaño, forma.
Cuando llega una fruta nueva, la clasifica.

## El código

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)        # Aprender fronteras
predictions = model.predict(X_test)  # Clasificar nuevos
```

## ¿Cómo decide?

1. Calcula probabilidad de cada clase
2. Asigna la clase con mayor probabilidad
3. Umbral: 0.5 por defecto

## Sigmoid

Convierte cualquier número en probabilidad (0-1):

```
σ(x) = 1 / (1 + e^(-x))
```

## Tipos de clasificación

| Tipo | Ejemplo |
|------|---------|
| **Binaria** | Spam / No spam |
| **Multiclase** | Gato / Perro / Pájaro |
| **Multilabel** | Etiquetas múltiples |

## Errores comunes

| Error | Solución |
|-------|----------|
| ConvergenceWarning | `max_iter=1000` |
| Accuracy bajo | Más features o modelo complejo |
| Clases desbalanceadas | `class_weight='balanced'` |
