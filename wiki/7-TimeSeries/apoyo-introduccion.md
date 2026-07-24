# Apoyo — Lección 1: Introducción a Series Temporales

## ¿Qué es una serie temporal?

Un dataset donde **el orden de las filas importa** porque están ordenadas por tiempo.

```
2012-01-01 00:00 → 2698 MW (carga eléctrica)
2012-01-01 01:00 → 2558 MW
2012-01-01 02:00 → 2444 MW
...
```

A diferencia de un dataset normal (ej. clientes), aquí **no puedes mezclar filas** sin perder información.

---

## El dataset: GEFCom2014

| Dato | Valor |
|------|-------|
| Período | Ene 2012 → Dic 2014 (3 años) |
| Frecuencia | Horaria (cada hora) |
| Total filas | 26,304 |
| Columnas | `load` (carga en MW), `temp` (temperatura en °F) |

**Target**: `load` — lo que queremos predecir (consumo eléctrico).

---

## Los 3 patrones que DEBES ver

### 1. Estacionalidad diaria (zig-zag cada 24h)
- **Noche**: carga baja (mundo durmiendo)
- **Mañana**: sube (fábricas, oficinas abren)
- **Tarde**: pico (colectivos, A/C, cocinas)
- **Noche**: baja de nuevo

Se ve en el zoom de julio 2014: el zig-zag se repite CADA DÍA.

### 2. Estacionalidad anual (zig-zag cada 12 meses)
- **Verano** (jul-ago): picos altos → mucho A/C
- **Invierno** (ene-feb): también sube → calefacción
- **Primavera/Otoño**: valles → clima amable, menos consumo

Se ve en la serie completa de 3 años.

### 3. Correlación carga ↔ temperatura
- **Correlación positiva** (≈ +0.6): cuando sube la temperatura, sube la carga
- Pero **NO es lineal perfecta**: hay días calurosos con carga baja (feriados, fines de semana)

---

## Código clave y qué significa

### Cargar datos
```python
from common.utils import load_data
df = load_data('./data')  # Descarga CSV si no existe
```

### Ver forma
```python
df.shape          # (26304, 2) → 26 mil filas, 2 columnas
df.columns        # ['load', 'temp']
df.dtypes         # load: float64, temp: float64
```

### Estadísticas rápidas
```python
df.describe()     # media, std, min, max, percentiles
```
Sirve para detectar outliers (valores extremos) y ver rangos.

### Valores faltantes
```python
df.isnull().sum()  # ¿Cuántos NaN hay?
```
Si hay muchos → hay que decidir: rellenar o eliminar.

### Visualizar serie completa
```python
df.plot(y='load', subplots=True, figsize=(15, 8))
```
**Qué buscar**: tendencia, picos anuales, gaps.

### Zoom en una semana
```python
df['2014-07-01':'2014-07-07'].plot(y='load')
```
**Qué buscar**: patrón diario (zig-zag cada 24h).

### Descomposición estacional
```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['load'], model='additive', period=24)
result.plot()
```
Separa en 3 partes:
- **Trend**: tendencia de largo plazo (sube/baja)
- **Seasonal**: patrón que se repite (cada 24h)
- **Resid**: lo que sobra (ruido, outliers)

---

## ¿Por qué importa esto para Lección 2?

| Lo que viste en Lección 1 | Cómo se usa en Lección 2 (ARIMA) |
|---------------------------|----------------------------------|
| Estacionalidad diaria (24h) | Parámetro `seasonal_period = 24` en SARIMA |
| La serie NO es estacionaria | Necesitas `d=1` o `d=2` (diferenciación) |
| Correlación carga ↔ temperatura | Puedes usar `temperature` como feature exógena |
| Outliers | Pueden romper el modelo → clean o usar robust solver |
| Walk-forward validation | **NUNCA** random split en series temporales |

---

## Conceptos que necesitas para Lección 2

### Estacionariedad
Una serie es **estacionaria** si su media y varianza **no cambian** con el tiempo.

```
✅ Estacionaria: fluctúa alrededor de ~3300 MW constante
❌ No-estacionaria: tiene tendencia clara hacia arriba/abajo
```

La mayoría de los modelos (incluyendo ARIMA) **asumen estacionariedad**. Si no lo es, hay que **diferenciar**.

### Diferenciación
Restar el valor anterior para "quitar" la tendencia:

```python
df['load_diff'] = df['load'] - df['load'].shift(1)
```

### Walk-forward validation
En series temporales **NUNCA** haces split random (como en classification).原因: el tiempo va en una dirección.

```
❌ Random split: Mezclar datos de 2012 con 2014 para train/test
✅ Walk-forward: Train = 2012-2013, Test = 2014
```

---

## Resumen rápido

```
Serie temporal = datos ordenados en el tiempo
                    ↓
¿Qué vemos? → Estacionalidad diaria + anual + correlación con temp
                    ↓
¿Qué hacemos? → Explorar, visualizar, descomponer
                    ↓
¿Para qué? → Para entender qué modelo usar en la Lección 2
```

---

**Siguiente paso**: [Lección 2: ARIMA](lesson-2-arima.md) — Construir el primer modelo de predicción.
