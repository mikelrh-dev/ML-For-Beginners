# Label Encoder 🔢

> "Los modelos solo entienden números."

## ¿Qué es?

Convierte categorías (texto) a números.
- "rojo" → 2
- "azul" → 0
- "verde" → 1

## Analogía

Como asignar un número a cada estudiante de la clase.
Ana=1, Luis=2, María=3.

## El código

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['color_codificado'] = le.fit_transform(df['color'])
```

## ¿Qué hace internamente?

1. `fit()` — Aprende las categorías únicas
2. `transform()` — Convierte a números
3. `fit_transform()` — Ambos en uno

## Ejemplo real

```python
# Antes
df['Country'] = ['US', 'GB', 'CA', 'US', 'GB']

# Después
df['Country'] = [3, 1, 0, 3, 1]
```

## ⚠️ Cuidado

LabelEncoder asume **orden ordinal**:
- 0 < 1 < 2
- Esto es correcto para: bajo, medio, alto
- Esto es INCORRECTO para: rojo, azul, verde

Para categorías sin orden, usa **OneHotEncoder**.

## OneHotEncoder vs LabelEncoder

| Tipo | Cuándo usar |
|------|-------------|
| **LabelEncoder** | Categorías ordinales (bajo, medio, alto) |
| **OneHotEncoder** | Categorías sin orden (rojo, azul, verde) |
