# Pickle 🥒

> "Serializar es como congelar una comida para después."

## ¿Qué es?

`pickle` serializa objetos Python: convierte un objeto en bytes que puedes guardar y cargar después.

## Analogía

Tu modelo entrenado es como una receta perfecta.
Pickle es como congelar ese plato para servirlo después.

## El código

```python
import pickle

# Guardar modelo
pickle.dump(model, open('modelo.p', 'wb'))

# Cargar modelo
model = pickle.load(open('modelo.p', 'rb'))
```

## Parámetros

| Modo | Significado | Uso |
|------|-------------|-----|
| `'wb'` | Write binary | Guardar |
| `'rb'` | Read binary | Cargar |

## ¿Qué se puede picklear?

- Modelos entrenados
- DataFrames
- Diccionarios
- Cualquier objeto Python

## ⚠️ Precauciones

1. **No confíes en pickles de fuentes desconocidas** — pueden ejecutar código malicioso
2. **Versión de Python** — un pickle de Python 3.6 puede no funcionar en 3.9
3. **Librerías** — el modelo necesita las mismas librerías instaladas

## Alternativas

| Formato | Ventaja |
|---------|---------|
| **pickle** | Simple, estándar |
| **joblib** | Mejor para numpy grandes |
| **ONNX** | Multiplataforma |
| **SavedModel** | TensorFlow/Keras |
