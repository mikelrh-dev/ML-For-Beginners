# Leccion 3: SVR — Modelo no-lineal para series temporales

> **ARIMA es lineal. SVR es flexible. Depende de tus datos cual funciona mejor.**

---

## Que aprendemos aqui?

1. **Que es SVR** y como funciona (Support Vector Regressor)
2. **Kernel RBF**: transformar datos a mayor dimension
3. **Timesteps**: crear ventanas de tiempo para el modelo
4. **Hiperparametros**: kernel, gamma, C, epsilon
5. **Comparacion**: SVR vs ARIMA

---

## El codigo (resumen)

```python
# 1. Crear modelo
model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)

# 2. Entrenar
model.fit(x_train, y_train[:,0])

# 3. Predecir
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

# 4. Invertir escala
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
```

---

## Desglose paso a paso

### 1. Timesteps

SVR necesita datos en forma **[batch, timesteps]**. En vez de usar 1 valor, usamos **5 valores pasados**:

```python
timesteps = 5

# Ventana de 5 valores
# [t-4, t-3, t-2, t-1, t] -> entrada: [t-4, t-3, t-2, t-1], salida: [t]

train_data_timesteps = np.array([
    [j for j in train_data[i:i+timesteps]] 
    for i in range(0, len(train_data)-timesteps+1)
])[:,:,0]
```

**Resultado:** Matriz de (N-4) filas x 5 columnas.

### 2. Separar inputs y outputs

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1], test_data_timesteps[:,[timesteps-1]]
```

- `x_train`: primeros 4 valores (inputs)
- `y_train`: ultimo valor (output)

### 3. Modelo SVR

```python
model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
model.fit(x_train, y_train[:,0])
```

| Parametro | Que controla | Nuestro valor |
|-----------|-------------|---------------|
| `kernel` | Tipo de curva | 'rbf' (flexible) |
| `gamma` | Influencia de cada punto | 0.5 |
| `C` | Penalizacion por error | 10 |
| `epsilon` | Margen de tolerancia | 0.05 |

### 4. Predecir y evaluar

```python
# Predecir
y_test_pred = model.predict(x_test).reshape(-1,1)

# Invertir escala
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test = scaler.inverse_transform(y_test)

# MAPE
print('MAPE:', mape(y_test_pred, y_test)*100, '%')
```

---

## SVR vs ARIMA

| Metrica | ARIMA | SVR |
|---------|-------|-----|
| Tipo | Lineal | No-lineal |
| Velocidad | Lenta (walk-forward) | Rapida |
| MAPE tipico | 0.5-1% | 1-2% |
| Interpretabilidad | Alta | Baja |
| Cuando usar | Serie estacionaria | Serie no-lineal |

---

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| Olvidar crear timesteps | Modelo no funciona |
| Timesteps muy grandes | Sobreajuste, lento |
| Timesteps muy pequenos | Modelo no aprende patrones |
| No invertir escala | MAPE en [0,1] no es interpretable |
| Olvidar sys.path.append | No encuentra common.utils |

---

## Notas tecnicas

- **Kernel RBF**: transforma datos a mayor dimension para encontrar patrones no-lineales
- **Support vectors**: puntos que definen la frontera del modelo
- **No necesita walk-forward**: SVR es rapido, se entrena una sola vez
- **timesteps=5**: balance entre capturar patron y velocidad

---

**Fin de la seccion TimeSeries.** Buen trabajo!
