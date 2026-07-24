# Apoyo: SVR (Support Vector Regressor)

> Resumen rapido para consultar mientras trabajas con el notebook.

---

## Que es SVR?

SVR es un modelo de **aprendizaje automatico** que predice valores futuros. A diferencia de ARIMA (que es estadistico), SVR usa **machine learning** para encontrar patrones en los datos.

---

## Flujo del notebook

1. **Cargar datos** → `energy.csv` con columna `load`
2. **Crear timesteps** → ventana de 5 valores pasados
3. **Separar train/test** → corte temporal
4. **Escalar** → MinMaxScaler [0,1]
5. **Crear modelo SVR** → kernel RBF
6. **Entrenar** → `model.fit(x_train, y_train)`
7. **Predecir** → `model.predict(x_test)`
8. **Invertir escala** → escala original
9. **Evaluar** → MAPE
10. **Comparar** → SVR vs ARIMA

---

## Que son los timesteps?

SVR necesita **ventanas de tiempo** en vez de un solo valor:

```
t=1: [10, 15, 12, 18, 20]  →  entrada: [10, 15, 12, 18]  →  salida: [20]
t=2: [15, 12, 18, 20, 22]  →  entrada: [15, 12, 18, 20]  →  salida: [22]
t=3: [12, 18, 20, 22, 19]  →  entrada: [12, 18, 20, 22]  →  salida: [19]
```

El numero de columnas en `x_train` es **timesteps - 1**.

---

## Hiperparametros

| Parametro | Que es | Que pasa si es... |
|-----------|--------|-------------------|
| `kernel` | Tipo de curva | 'rbf' es flexible |
| `gamma` | Influencia de cada punto | Alto = sobreajuste |
| `C` | Penalizacion por error | Alto = mas estricto |
| `epsilon` | Margen de tolerancia | Alto = modelo mas suave |

---

## SVR vs ARIMA: cuando usar cual?

| Situacion | Mejor modelo |
|-----------|-------------|
| Serie lineal, sin estacionalidad | ARIMA |
| Serie con patrones complejos | SVR |
| Necesitas rapidez | SVR |
| Necesitas explicabilidad | ARIMA |
| Datos estacionarios | ARIMA |

---

## Codigo esencial

```python
# Crear modelo
from sklearn.svm import SVR
model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)

# Entrenar
model.fit(x_train, y_train[:,0])

# Predecir
y_test_pred = model.predict(x_test).reshape(-1,1)
```

---

**Fin de la seccion TimeSeries.**
