# Leccion 2: ARIMA — Nuestro primer modelo de prediccion

> **No necesitas entender toda la matematica. Necesitas entender que hace, por que lo hace, y como medir si funciona.**

---

## Que aprendemos aqui?

1. **Que es ARIMA** y como funciona (AR + I + MA)
2. **SARIMAX**: ARIMA con componente estacional
3. **Split temporal**: por que NUNCA random split en series temporales
4. **MinMaxScaler**: escalar datos al rango [0,1]
5. **Walk-forward validation**: re-entrenar con cada dato nuevo
6. **MAPE**: medir calidad de prediccion

---

## El codigo (resumen)

```python
# 1. Cargar y escalar
energy = load_data('./data')[['load']]
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
test['load'] = scaler.transform(test)

# 2. Configurar SARIMAX
order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)
model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
results = model.fit()

# 3. Walk-forward
for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=HORIZON)
    # ... actualizar historial
```

---

## Desglose paso a paso

### 1. Split temporal

```python
train_start_dt = '2014-11-01 00:00:00'
test_start_dt = '2014-12-30 00:00:00'
```

**Por que no random split?** En series temporales, el tiempo va en una direccion. Si mezclas datos de 2012 con 2014 en el train, el modelo "ve" el futuro y hace trampa.

```
Train: Nov 1 -> Dic 29, 2014  (1416 horas)
Test:  Dic 30 -> Dic 31, 2014 (48 horas)
```

### 2. Escalar con MinMaxScaler

```python
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)  # fit + transform
test['load'] = scaler.transform(test)        # solo transform
```

**Regla de oro:** `fit_transform()` en train, `transform()` en test. Si haces fit en test, el scaler "ve" los valores futuros = trampa.

### 3. Configurar SARIMAX

```python
order = (4, 1, 0)              # p, d, q
seasonal_order = (1, 1, 0, 24) # P, D, Q, m
```

| Parametro | Que controla | Nuestro valor | Por que |
|-----------|-------------|---------------|---------|
| `p` | Lags auto-regresivos | 4 | Mira 4 horas atras |
| `d` | Diferenciacion | 1 | Quita tendencia con 1 resta |
| `q` | Media movil | 0 | No usa errores pasados |
| `P` | Lags estacionales | 1 | Mira valor de hace 24h |
| `D` | Dif. estacional | 1 | Quita tendencia estacional |
| `Q` | MA estacional | 0 | No usa errores estacionales |
| `m` | Periodo | 24 | Patron se repite cada 24h |

### 4. Walk-forward validation

```python
training_window = 720  # 30 dias

for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=HORIZON)
    predictions.append(yhat)
    # Mover ventana
    history.append(obs[0])
    history.pop(0)
```

**Que es walk-forward?** Re-entrenar el modelo con cada nuevo dato. Simula produccion real.

### 5. Evaluar con MAPE

```python
# One-step (solo t+1)
one_step_mape = mape(eval_df[eval_df['h'] == 't+1']['prediction'],
                     eval_df[eval_df['h'] == 't+1']['actual']) * 100

# Multi-step (todos los horizontes)
multi_step_mape = mape(eval_df['prediction'], eval_df['actual']) * 100
```

| MAPE | Que significa |
|------|---------------|
| < 1% | Excelente |
| 1-2% | Muy bueno |
| 2-5% | Bueno |
| 5-10% | Regular |
| > 10% | Malo |

---

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| Random split en vez de temporal | Modelo "ve" el futuro = trampa |
| fit_transform en test | Scaler "ve" valores futuros |
| Ventana de entrenamiento muy chica | Modelo no aprende patrones |
| Ventana muy grande | Walk-forward es muy lento |
| Olvidar invertir escala | MAPE en [0,1] no es interpretable |

---

## Notas tecnicas

- **Ventana de entrenamiento**: 720 horas (30 dias) — balance entre velocidad y calidad
- **HORIZON**: 3 horas — cuantos pasos adelante predecimos
- **SARIMAX vs ARIMA**: SARIMAX agrega componente estacional (S) y features exogenas (X)
- **statsmodels**: libreria que implementa SARIMAX en Python

---

**Siguiente:** [Leccion 3: SVR](lesson-3-svr.md) — Modelo no-lineal alternativo
