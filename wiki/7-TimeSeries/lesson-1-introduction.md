# Lección 1: Introducción a Series Temporales

> **Antes de predecir, observa. Conocer los datos es la mitad del trabajo.**

---

## ¿Qué aprendemos aquí?

1. **Qué es una serie temporal** y qué la hace especial
2. **Patrones comunes**: tendencia, estacionalidad, outliers
3. **Cómo explorar** datos temporales: cargar, visualizar, descomponer
4. **El dataset GEFCom2014**: 3 años de consumo eléctrico horario

---

## El código

```python
import pandas as pd
import matplotlib.pyplot as plt
from common.utils import load_data

df = load_data('./data')
df.plot(y='load', subplots=True, figsize=(15, 8))
plt.show()
```

---

## Desglose paso a paso

### 1. Cargar los datos

```python
df = load_data('./data')
```

La función `load_data()` viene de `common/utils.py`. Descarga el CSV si no existe y devuelve el DataFrame con el índice ya configurado como datetime.

### 2. Explorar la forma

```python
print(df.shape)        # (26304, 3)
print(df.columns)      # ['load', 'temperature']
print(df.index.min())  # 2012-01-01 00:00:00
print(df.index.max())  # 2014-12-31 23:00:00
```

**Datos esperados:**
- Frecuencia: horaria
- Período: 2012-01-01 → 2014-12-31 (3 años × 365 días × 24h ≈ 26,280 filas)
- Columnas: `load` (MW), `temperature` (°C)

### 3. Visualizar la serie completa

```python
df.plot(y='load', subplots=True, figsize=(15, 8))
```

**Qué buscar visualmente:**

| Patrón | Apariencia |
|--------|-----------|
| **Tendencia** | Pendiente clara (sube, baja) |
| **Estacionalidad** | Zig-zag repetitivo (días, semanas, años) |
| **Outliers** | Picos alejados del nivel normal |
| **Huecos** | Gaps donde faltan datos |
| **Cambios de régimen** | Saltos bruscos permanentes |

### 4. Zoom en una semana

```python
df['2014-07-01':'2014-07-07'].plot(y='load', figsize=(15, 6))
```

Revela el **patrón diario**: picos durante el día, valles durante la noche.

### 5. Descomposición

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['load'], model='additive', period=24)
result.plot()
```

Separa la serie en:
- **Tendencia** (movimiento de largo plazo)
- **Estacionalidad** (patrón periódico, ej. cada 24h)
- **Residual** (lo que sobra, ruido + outliers)

---

## ¿Por qué esto?

**Sin exploración, no hay modelo bueno**.

Antes de elegir ARIMA, SVR o lo que sea, necesitas saber:

| Pregunta | Por qué importa |
|----------|-----------------|
| ¿Hay tendencia? | Determina si necesitas diferenciación |
| ¿Hay estacionalidad? | Determina el parámetro `period` (SARIMAX) |
| ¿Hay outliers? | Decide solver robusto o limpieza previa |
| ¿Cuál es el target? | `load` (MW) — lo que queremos predecir |
| ¿Hay features exógenas? | `temperature` puede mejorar el modelo |

---

## Conceptos clave

| Concepto | Definición |
|----------|------------|
| **Serie temporal** | Datos ordenados temporalmente |
| **Índice Datetime** | Pandas index = timestamps |
| **Tendencia** | Dirección a largo plazo |
| **Estacionalidad** | Patrón que se repite cada N pasos |
| **Outlier** | Valor atípico |
| **Estacionariedad** | Media y varianza constantes en el tiempo |
| **Descomposición** | Separar tendencia + estacionalidad + ruido |
| **Target** | Lo que queremos predecir (`load`) |
| **Feature exógena** | Variable externa que ayuda (ej. `temperature`) |

---

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| Asumir que los datos están limpios | Outliers rompen el modelo |
| No graficar la serie completa | No ver tendencia de fondo |
| Trasponer datos por accidente | Se pierde la propiedad temporal |
| Olvidar que el índice ES la variable predictora | Pierdes el tiempo como feature |

---

## ¿Qué sigue?

[**Lección 2: ARIMA**](lesson-2-arima.md) → Construir el primer modelo de predicción.

[**Lección 3: SVR**](lesson-3-svr.md) → Alternativa no-linear para series temporales.

---

## Notas técnicas

- Frecuencia: **horaria** (1h). Para convertir a diaria: `df.resample('D').mean()`
- Si quieres predecir en **intervalos regulares**, usa `.resample()`
- Para validar: walk-forward (no random split)
