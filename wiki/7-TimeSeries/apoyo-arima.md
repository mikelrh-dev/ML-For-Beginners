# Apoyo — Leccion 2: ARIMA

## Resumen rapido

ARIMA es un modelo que aprende de **3 fuentes** para predecir series temporales:
1. **Pasado** (AR): lo que paso antes predice el futuro
2. **Tendencia** (I): quitar la direccion para estabilizar
3. **Errores** (MA): corregir lo que me equivoque

---

## Los 6 parametros que DEBES entender

### Componente no-estacional: `order = (p, d, q)`

| Parametro | Que hace | Ejemplo | Como elegirlo |
|-----------|----------|---------|---------------|
| `p` | Cuantos valores pasados mira | 4 = mira 4 horas atras | Autocorrelacion (ACF) |
| `d` | Cuantas veces restar el anterior | 1 = una vez | Hasta que la serie sea estacionaria |
| `q` | Cuantos errores pasados usa | 0 = ninguno | Autocorrelacion parcial (PACF) |

### Componente estacional: `seasonal_order = (P, D, Q, m)`

| Parametro | Que hace | Ejemplo | Como elegirlo |
|-----------|----------|---------|---------------|
| `P` | Lags estacionales | 1 = mira hace 24h | ACF en multiples de m |
| `D` | Dif. estacional | 1 = una vez | Hasta que la estacion sea estable |
| `Q` | MA estacional | 0 = ninguno | PACF en multiples de m |
| `m` | Periodo | 24 = cada 24h | Conocimiento del dominio |

### Nuestra configuracion

```
order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)
```

**Por que p=4?** La hora pasada es la mejor prediccion de la hora futura.
**Por que d=1?** La serie tiene leve tendencia, la quitamos con 1 resta.
**Por que q=0?** No usamos correccion por errores (AR puro).
**Por que P=1?** Mira el valor de hace 24 horas (patron diario).
**Por que m=24?** El patron se repite cada 24 horas.

---

## Split temporal: POR QUE no random split

```
Si hago random split:
  Train: [2014-03-15, 2012-07-20, 2014-11-01, ...]  (mezclado)
  Test:  [2014-12-30, 2012-09-10, 2014-06-15, ...]  (mezclado)

El modelo "ve" el futuro y hace TRAMPA!
```

```
Si hago split temporal:
  Train: [2014-11-01, 2014-11-02, ..., 2014-12-29]  (ordenado)
  Test:  [2014-12-30, 2014-12-31]  (despues del train)

El modelo NUNCA ve el futuro. Justo como en la vida real.
```

---

## MinMaxScaler: POR QUE escalar

```
original = 3000 MW

escalado = (original - min) / (max - min)
         = (3000 - 1979) / (5224 - 1979)
         = 1021 / 3245
         = 0.315
```

**Regla de oro:**
- `fit_transform()` en train: aprende el rango y escala
- `transform()` en test: usa el mismo rango del train

Si haces `fit_transform()` en test, el scaler "ve" los valores futuros = trampa.

---

## Walk-forward: POR QUE re-entrenar

```
Paso 1: Entrena en [horas 1-720]  -> Predice hora 721
Paso 2: Entrena en [horas 2-721]  -> Predice hora 722
Paso 3: Entrena en [horas 3-722]  -> Predice hora 723
...
```

**Por que no simplemente predecir todo de una?**

Porque en la vida real, cada hora llega un dato nuevo y queremos la mejor prediccion posible con la informacion mas reciente.

**Ventana de 720 horas (30 dias):**
- Datos muy viejos pueden no ser relevantes
- Mas datos = mas tiempo de entrenamiento
- 30 dias es un buen balance

---

## MAPE: como medir el error

```
MAPE = mean( |real - predicho| / real ) x 100%
```

| MAPE | Que significa |
|------|---------------|
| 0.5% | Excelente |
| 2% | Bueno |
| 5% | Regular |
| 10% | Malo |

**Interpretacion:** Si MAPE = 2%, mis predicciones tienen un error promedio del 2%.

---

## Que aprendimos para Leccion 3

| Lo que viste en Leccion 2 | Como se usa en Leccion 3 (SVR) |
|---------------------------|-------------------------------|
| Split temporal | Mismo concepto, diferente modelo |
| MinMaxScaler | SVR tambien necesita escalado |
| Walk-forward | Mismo metodo de validacion |
| MAPE | Misma metrica de evaluacion |
| SARIMAX | SVR es una alternativa no-lineal |

---

## Preguntas frecuentes

**P: Por que usamos SARIMAX y no ARIMA simple?**
R: Porque nuestra serie tiene estacionalidad (patron que se repite cada 24h). SARIMAX agrega componente estacional.

**P: Que pasa si cambio los parametros?**
R: El modelo puede mejorar o empeorar. Hay que experimentar (grid search).

**P: Walk-forward es lento, hay otra forma?**
R: Si, puedes usar un solo train/test split. Pero walk-forward es mas realista.

**P: Que es HORIZON?**
R: Cuantos pasos adelante predecimos. Con HORIZON=3, predecimos las proximas 3 horas.

---

**Siguiente paso:** [Leccion 3: SVR](lesson-3-svr.md) — Comparar con un modelo no-lineal.
