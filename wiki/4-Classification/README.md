# Clasificación — Poniendo categorías a los datos 🏷️

> "La clasificación es como organizar tu ropero: cada cosa tiene su lugar."

## ¿Qué es la clasificación?

La clasificación es predecir a qué **categoría** pertenece un dato.
A diferencia de la regresión (que predice números), acá predicciones son etiquetas.

## Tipos de clasificación

| Tipo | Ejemplo | Salida |
|------|---------|--------|
| **Binaria** | ¿Este email es spam? | Sí / No |
| **Multiclase** | ¿De qué cocina es esta receta? | Tailandés / Japonés / Chino / Indio / Coreano |

## Lecciones en esta sección

1. **1-Introduction** - Preparar datos y balancear con SMOTE
2. **2-Classifiers-1** - Logistic Regression y métricas
3. **3-Classifiers-2** - Comparar múltiples algoritmos
4. **4-Applied** - App de recomendación con ONNX

## Dataset

Usaremos recetas de cocinas asiáticas 🍜 — 5 cocinas, 384 ingredientes

## Flujo de trabajo

```
Datos crudos → Limpiar → Balancear (SMOTE) → Dividir → Entrenar → Evaluar
```
