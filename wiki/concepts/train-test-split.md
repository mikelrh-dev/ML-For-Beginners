# Train-Test Split 🎲

> "Nunca evalúes tu modelo con los mismos datos con los que lo entrenaste."

## ¿Qué es?

Dividir tus datos en dos partes:
- **Training set** (80%): Para entrenar el modelo
- **Test set** (20%): Para evaluar el modelo

## Analogía

Es como estudiar para un examen con ejercicios y luego hacer un examen nuevo.
Si solo practicas los mismos ejercicios, no sabes si realmente aprendiste.

## El código

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para testing
    random_state=42     # Reproducibilidad
)
```

## ¿Por qué `random_state`?

Sin él, cada vez que ejecutas obtienes divisiones diferentes.
Con `random_state=42`, siempre obtienes la misma división.

## Errores comunes

| Error | Solución |
|-------|----------|
| Test set muy grande | Usar `test_size=0.2` o `0.3` |
| Test set muy pequeño | Usar `test_size=0.3` o más |
| Datos desbalanceados | Usar `stratify=y` |
