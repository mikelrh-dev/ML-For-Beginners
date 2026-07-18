# 4. Regresión Logística 🏷️

> "No todo es blanco o negro... pero el modelo intenta decidir."

## ¿Qué aprendemos aquí?

La regresión logística predice **categorías**, no números.
Como clasificar emails: spam o no spam.

## El código

```python
from sklearn.linear_model import LogisticRegression

# Crear modelo
model = LogisticRegression()

# Entrenar
model.fit(X_train, y_train)

# Predecir
predictions = model.predict(X_test)

# Evaluar
accuracy = model.score(X_test, y_test)
print(f'Precisión: {accuracy:.2%}')
```

## Desglose paso a paso

| Paso | Código | ¿Qué hace? |
|------|--------|-------------|
| Crear | `LogisticRegression()` | Modelo de clasificación |
| Entrenar | `model.fit(X, y)` | Aprende fronteras |
| Predecir | `model.predict()` | Asigna categoría |
| Evaluar | `model.score()` | % de aciertos |

## ¿Por qué esto?

1. **Clasificación**: Predecir categorías (0 o 1, rojo o azul)
2. **Probabilidades**: Internamente calcula probabilidad de cada clase
3. **Sigmoid**: Convierte cualquier número en probabilidad (0-1)

## Conceptos clave

- **LogisticRegression**: Clasificador binario/multiclase
- **Clases**: Categorías (spam/no spam, aprobado/rechazado)
- **Accuracy**: Porcentaje de predicciones correctas
- **Sigmoid**: Función que aplana números a probabilidades

## Errores comunes

```python
# ❌ ConvergenceWarning
# Modelo no convergió
# ✅ Aumentar max_iter=1000

# ❌ Accuracy bajo (< 60%)
# Datos no separables o modelo simple
# ✅ Probar más features o modelo complejo
```

## Siguiente paso

Ya sabes regresión lineal y logística. Vamos a ver cómo guardar modelos → [Glossary](./glossary.md)
