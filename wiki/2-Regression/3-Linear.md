# 3. Regresión Lineal 📈

> "El patrón más simple en la naturaleza es una línea recta."

## ¿Qué aprendemos aquí?

La regresión lineal encuentra la mejor línea que conecta tus datos.
Como encontrar la tendencia de una gráfica.

## El código

```python
from sklearn.linear_model import LinearRegression

# Crear modelo
model = LinearRegression()

# Entrenar
model.fit(X_train, y_train)

# Predecir
predictions = model.predict(X_test)

# Evaluar
score = model.score(X_test, y_test)
print(f'R² score: {score:.2f}')
```

## Desglose paso a paso

| Paso | Código | ¿Qué hace? |
|------|--------|-------------|
| Crear | `LinearRegression()` | Instanciar el modelo |
| Entrenar | `model.fit(X, y)` | Aprender patrones |
| Predecir | `model.predict(X_test)` | Generar predicciones |
| Evaluar | `model.score()` | Medir precisión (R²) |

## ¿Por qué esto?

1. **R² score**: Mide qué tan bien la línea explica los datos (0-1)
2. **Coeficientes**: La pendiente e intersección de la línea
3. **Limitaciones**: Solo funciona con relaciones lineales

## Conceptos clave

- **LinearRegression**: Modelo que encuentra la mejor línea
- **fit()**: Entrenar el modelo con datos
- **predict()**: Usar el modelo para predecir
- **R² score**: Métrica de precisión (1.0 = perfecto)

## Errores comunes

```python
# ❌ ValueError: could not convert string to float
# Categorías no codificadas
# ✅ Usar LabelEncoder primero

# ❌ R² score muy bajo (< 0.5)
# Relación no es lineal
# ✅ Probar PolynomialFeatures
```

## Siguiente paso

La regresión lineal predice números. ¿Qué pasa si quieres predecir categorías? → [4-Logistic](./4-Logistic.md)
