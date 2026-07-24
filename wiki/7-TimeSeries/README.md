# 7. TimeSeries — Predicción de Series Temporales

## ¿Qué son las series temporales?

Datos **ordenados en el tiempo**. Ejemplos:

- Precio de una acción cada día
- Carga eléctrica cada hora
- Ventas mensuales de un producto
- Temperatura cada minuto

**La clave**: el **orden importa**. A diferencia de otros datos, no podemos mezclar filas sin perder información.

## ¿Por qué es importante?

| Aplicación | Caso de uso |
|------------|-------------|
| Energía | Predecir demanda eléctrica para balancear la red |
| Finanzas | Predecir precio de acciones, riesgo |
| Retail | Predecir ventas para optimizar inventario |
| Meteorología | Temperatura, lluvia, viento |
| Logística | Carga de servidores, demanda de transporte |

## Conceptos clave

| Concepto | Definición | Analogía |
|----------|------------|----------|
| **Tendencia** | Dirección a largo plazo (sube, baja) | La pendiente del gráfico |
| **Estacionalidad** | Patrón que se repite cada N período | "Cada diciembre hay pico de ventas" |
| **Estacionariedad** | Media y varianza constantes en el tiempo | "Datos estables, sin tendencia" |
| **Diferenciación** | Restar valor anterior para quitar tendencia | "Estabilizar la serie" |
| **ARIMA** | Modelo clásico: AutoRegresivo + Integrado + MA | "Aprende de lags + errores pasados" |

## Lecciones

1. **[Introducción a Series Temporales](lesson-1-introduction.md)** — EDA, visualización, identificar patrones
   - [Apoyo Lección 1](apoyo-introduccion.md) — Puntos clave para entender
2. **[ARIMA](lesson-2-arima.md)** — Modelo clásico para series estacionarias
   - [Apoyo Lección 2](apoyo-arima.md) — Puntos clave para entender
3. **[SVR — Support Vector Regressor](lesson-3-svr.md)** — Regresión no-lineal para series temporales

## Glosario

- [Glosario completo](glossary.md) — Todos los términos de TimeSeries

## Datasets usados

| Dataset | Origen | Descripción |
|---------|--------|-------------|
| **GEFCom2014** | Global Energy Forecasting Competition | 3 años de carga eléctrica horaria + temperatura (2012-2014) |

---

**Siguiente sección:** [8-Reinforcement](../8-Reinforcement/README.md) — Aprendizaje por refuerzo*
