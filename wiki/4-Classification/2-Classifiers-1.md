# 2. Logistic Regression y métricas 📊

> "No alcanza con saber si el modelo acierta — hay que saber CÓMO acierta."

## ¿Qué aprendemos aquí?

Entrenamos nuestro primer clasificador y aprendemos a evaluarlo
con métricas más detalladas que solo accuracy.

## El código

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Entrenar
lr = LogisticRegression(multi_class='ovr', solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

# Evaluar
accuracy = model.score(X_test, y_test)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Desglose paso a paso

| Paso | Código | ¿Qué hace? |
|------|--------|-------------|
| Dividir | `train_test_split()` | 70% entrenar, 30% evaluar |
| Crear modelo | `LogisticRegression()` | Clasificador multiclase |
| Entrenar | `model.fit()` | Aprende patrones |
| Evaluar | `model.score()` | Accuracy general |
| Reporte | `classification_report()` | Métricas por clase |

## Métricas explicadas

| Métrica | Pregunta que responde | Ejemplo |
|---------|----------------------|---------|
| **Accuracy** | ¿Cuántos aciertos en total? | 80 de 100 → 80% |
| **Precision** | De los que dije "indian", ¿cuántos lo eran? | 90 de 100 → 90% |
| **Recall** | De los que ERAN "indian", ¿cuántos detecté? | 85 de 100 → 85% |
| **F1-score** | Balance precision-recall | Media armónica |

## ¿Qué es un solver?

El solver es el algoritmo de optimización. No es el modelo — es CÓMO el modelo encuentra los mejores coeficientes.

| Solver | Mejor para |
|--------|------------|
| `liblinear` | Datasets pequeños |
| `lbfgs` | Multiclass, default |

## Conceptos clave

- **OvR**: One-vs-Rest — un clasificador binario por cada clase
- **Classification report**: Métricas detalladas por clase
- **Confusion matrix**: Tabla de aciertos y errores

## Errores comunes

| Error | Solución |
|-------|----------|
| ConvergenceWarning | `max_iter=1000` |
| Accuracy bajo | Probar otro solver o algoritmo |
| Clases desbalanceadas | SMOTE en el paso anterior |
