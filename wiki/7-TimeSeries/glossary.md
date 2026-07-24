# Glosario de TimeSeries — Series Temporales

## Términos fundamentales

### Serie Temporal (Time Series)
Datos **ordenados en el tiempo**, generalmente con intervalos regulares.

```
[10:00] → 2,500 MW
[11:00] → 2,700 MW
[12:00] → 3,100 MW
[13:00] → 3,200 MW
```

El **orden importa** — no es un dataset tabular típico.

### Índice Datetime
Pandas index que almacena fechas/horas. Permite:

```python
df.loc['2014-07-01']               # Filtrar por fecha
df['2014-07-01':'2014-07-07']      # Rango
df.resample('D').mean()            # Cambiar frecuencia (hora → día)
```

---

## Patrones

### Tendencia (Trend)
Dirección de largo plazo. Sube, baja, o se mantiene.

```
↑ Tendencia creciente: ventas anuales pasando de 100 → 200 → 300
```

### Estacionalidad (Seasonality)
Patrón que se **repite cada N períodos**.

| Frecuencia | Ejemplo |
|------------|---------|
| Horaria (24) | Carga eléctrica baja de noche, alta de día |
| Diaria (7) | Ventas bajas los domingos |
| Anual (12) | Pico de ventas en diciembre |

### Outlier (Valor atípico)
Valor lejano al nivel normal.

```
Pico de 10,000 MW un martes a las 3am → probablemente sensor roto
```

### Ciclo a largo plazo
Tendencia secundaria que dura **más de una estación**.

Ej: recesiones económicas cada 7-10 años.

### Cambio abrupto (Regime change)
Salto permanente en el comportamiento de la serie.

```
COVID-19 → caída permanente del consumo eléctrico en marzo 2020
```

---

## Estacionariedad y diferenciación

### Estacionariedad
Serie cuya **media y varianza no cambian** en el tiempo.

```
✅ Estacionaria: fluctuación constante alrededor de ~2500 MW
❌ No-estacionaria: tendencia clara hacia arriba
```

### Diferenciación
Restar el valor anterior para eliminar tendencia.

```python
df['load_diff'] = df['load'] - df['load'].shift(1)
```

`shift(1)` es el "rezago" (lag) de 1 período.

### Diferenciación estacional
Restar el valor de la misma estación anterior.

```python
df['load_diff_24'] = df['load'] - df['load'].shift(24)  # Para estacionalidad horaria
```

### Test ADF (Augmented Dickey-Fuller)
Test estadístico para comprobar estacionariedad.

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['load'])
print(f"p-valor: {result[1]:.4f}")
# p < 0.05 → estacionaria
```

---

## Modelos

### ARIMA (AutoRegressive Integrated Moving Average)
Modelo clásico de pronóstico. Tiene 3 parámetros: `p`, `d`, `q`.

### AR (AutoRegresivo)
El valor actual depende de **valores pasados**.

```
Y_t = α₁·Y_(t-1) + α₂·Y_(t-2) + ruido
```

### I (Integrado / Diferenciación)
Cuántas veces diferenciamos la serie para hacerla estacionaria.

### MA (Media Móvil)
El valor actual depende de **errores pasados**.

```
Y_t = β₁·error_(t-1) + β₂·error_(t-2) + ruido
```

### ARIMA(p, d, q)
- `p`: lags auto-regresivos
- `d`: diferencias aplicadas
- `q`: lags de media móvil

### SARIMA(p, d, q)(P, D, Q, m)
ARIMA + **componente estacional**.

`(P, D, Q, m)` describe el componente estacional, donde `m` es el período (24 para diario, 12 para anual).

### SARIMAX
SARIMA + **features exógenas** (X). Permite usar variables externas (ej. temperatura) para mejorar la predicción.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(endog=train, order=(4,1,0), seasonal_order=(1,1,0,24))
```

### MinMaxScaler
Escala los datos al rango **[0, 1]**. ARIMA funciona mejor con datos acotados.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)  # fit + transform
test['load'] = scaler.transform(test)        # solo transform
```

**Regla de oro:** `fit_transform()` en train, `transform()` en test.

### Ventana de entrenamiento (Training Window)
Cantidad de datos pasados que usa el modelo para entrenar. En nuestro caso: **720 horas (30 días)**.

```python
training_window = 720  # 30 días
history = history[(-training_window):]  # últimos 720 valores
```

- Ventana muy chica → modelo no aprende patrones
- Ventana muy grande → walk-forward es muy lento

---

## SVR (Support Vector Regressor)

Modelo de **Support Vector Machine** aplicado a regresión. Bueno para capturar **no-linealidad** en series temporales.

```python
from sklearn.svm import SVR
model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
```

### Kernel
Función que transforma los datos a mayor dimensión. Comunes:
- `'linear'`: problemas lineales
- `'rbf'`: no-lineal, Gaussian Radial Basis Function
- `'poly'`: polinomial

### Timesteps (Pasos temporales)
Número de valores pasados que usa el modelo como entrada.

Con `timesteps = 5`:
```
[t-4, t-3, t-2, t-1, t-0] → predecir [t+1]
Entradas (4):    Y_(t-4), Y_(t-3), Y_(t-2), Y_(t-1)
Target (1):      Y_t
```

---

## Validación

### Walk-forward Validation
Validación **temporal**: re-entrenar el modelo con cada nuevo dato.

```
t=1: Entrena en [datos históricos] → Predice [t+1]
t=2: Entrena en [datos históricos + t+1] → Predice [t+2]
...
```

**Nunca** hacer random split en series temporales → fuga de datos.

### Train/Test Split temporal
Cortar la serie en el tiempo:
- **Train**: primeros 80% (ej. sept-oct 2014)
- **Test**: últimos 20% (ej. nov-dic 2014)

---

## Métricas

### MAPE (Mean Absolute Percentage Error)
Error relativo promedio. **En porcentaje**.

```
MAPE = mean( |actual - predicho| / actual ) × 100%
```

- `MAPE = 2%` → muy preciso
- `MAPE > 10%` → malo

### MAE (Mean Absolute Error)
Error absoluto medio. **En unidades del target**.

```
MAE = mean( |actual - predicho| )
```

### RMSE (Root Mean Squared Error)
Raíz del error cuadrático medio. Penaliza más los errores grandes.

---

## Conceptos avanzados

### Forecasting Horizon
Cuántos pasos adelante quieres predecir.

```python
HORIZON = 24  # predecir 24 horas
```

- **One-step**: predices `t+1`
- **Multi-step**: predices `t+1`, `t+2`, ..., `t+h`

### Lag (Rezago)
Valor publicado N períodos atrás.

```python
df['load_lag1'] = df['load'].shift(1)   # 1 período atrás
df['load_lag24'] = df['load'].shift(24) # 24 períodos atrás (1 día)
```

### Autocorrelación (ACF)
Correlación de la serie con sus propios lags.

```python
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['load'])
```

Picos en lags múltiplos de 24 → estacionalidad diaria.

---

---

**Volver al [índice de TimeSeries](README.md)**
