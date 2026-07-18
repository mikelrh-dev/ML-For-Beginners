# 1. Preparando datos para clasificación 🧹

> "Datos limpios producen modelos limpios."

## ¿Qué aprendemos aquí?

Antes de entrenar, hay que limpiar y **balancear** el dataset.
Si hay más recetas indias que tailandesas, el modelo se va a sesgar.

## El código

```python
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Cargar datos
df = pd.read_csv('../data/cuisines.csv')

# Quitar ingredientes comunes (rice, garlic, ginger)
feature_df = df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine

# Balancear con SMOTE
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)

# Guardar
transformed_df = pd.concat([transformed_label_df, transformed_feature_df], axis=1)
transformed_df.to_csv('../data/cleaned_cuisines.csv')
```

## Desglose paso a paso

| Paso | Código | ¿Por qué? |
|------|--------|------------|
| Cargar | `read_csv()` | Leer el dataset |
| Quitar columnas | `drop()` | Ingredientes comunes no ayudan a distinguir |
| SMOTE | `fit_resample()` | Balancear clases desbalanceadas |
| Guardar | `to_csv()` | Dataset listo para entrenar |

## ¿Por qué SMOTE?

Si tenés 799 recetas coreanas y solo 289 tailandesas, el modelo va a aprender a predecir "coreano" porque es más probable estadísticamente. SMOTE genera ejemplos sintéticos de las cocinas con menos datos para igualar todas a 799.

## Conceptos clave

- **SMOTE**: Genera ejemplos sintéticos de minorías
- **Balanceo**: Igualar la cantidad de ejemplos por clase
- **Features**: Ingredientes (columnas con 0 o 1)
- **Label**: Cocina de origen (lo que queremos predecir)

## Errores comunes

| Error | Solución |
|-------|----------|
| Datos desbalanceados | SMOTE o `class_weight='balanced'` |
| Columnas innecesarias | `drop(['Unnamed: 0'])` |
| Feature leakage | No incluir el label en las features |
