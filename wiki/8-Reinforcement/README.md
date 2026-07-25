# 8. Reinforcement — Aprendizaje por Refuerzo

## ¿Qué es el Aprendizaje por Refuerzo?

Un **agente** aprende a tomar **acciones** en un **entorno** para maximizar una **recompensa**. A diferencia de supervisado (clasificación/regresión), aquí **no hay datos etiquetados** — el agente aprende probando, fallando y recibiendo retroalimentación.

**La analogía clásica**: entrenar a un perro. Cuando hace algo bien (sentarse), recibe una galleta (recompensa positiva). Cuando hace algo mal, no recibe nada o recibe un "no" (recompensa negativa). Con repetición, el perro aprende qué acciones producen la galleta.

## ¿Por qué es importante?

| Aplicación | Caso de uso |
|------------|-------------|
| Robótica | Aprender a caminar, agarrar objetos |
| Juegos | AlphaGo, Dota 2, ajedrez |
| Vehículos autónomos | Conducir, estacionar, evitar obstáculos |
| Finanzas | Trading algorítmico, gestión de cartera |
| Control industrial | Optimizar procesos en fábricas |

## Conceptos clave

| Concepto | Definición | Analogía |
|----------|------------|----------|
| **Agente** | Quien toma las decisiones | Pedro (el lobo lo persigue) |
| **Entorno** | El mundo donde el agente opera | El tablero 8x8 con obstáculos |
| **Estado** | Situación actual del agente | Coordenadas (x,y) de Pedro |
| **Acción** | Lo que el agente puede hacer | Arriba, Abajo, Izquierda, Derecha |
| **Recompensa** | Feedback numérico por cada acción | +10 por manzana, -10 por lobo |
| **Política** | Estrategia que mapea estado → acción | "En (3,4) siempre mover derecha" |
| **Q-Table** | Tabla que puntúa cada acción por estado | "En (3,4) → derecha vale 0.8, arriba 0.2" |
| **Alpha (α)** | Tasa de aprendizaje | Qué tan rápido olvida lo viejo por lo nuevo |
| **Gamma (γ)** | Factor de descuento | Cuánto importa la recompensa futura vs inmediata |

## Lecciones

1. **[Q-Learning: Introducción a RL](lesson-1-qlearning.md)** — El mundo de Pedro, Q-Table, Bellman, exploración vs explotación
   - [Apoyo Lección 1](apoyo-qlearning.md) — Puntos clave para entender
2. **[CartPole con Gym](lesson-2-gym.md)** — Estado continuo, discretización, OpenAI Gym
   - [Apoyo Lección 2](apoyo-gym.md) — Puntos clave para entender

## Glosario

- [Glosario completo](glossary.md) — Todos los términos de Reinforcement Learning

## Entornos usados

| Entorno | Descripción | Tipo de estado |
|---------|-------------|----------------|
| **Peter & Wolf** | Tablero 8x8 con manzanas, lobo y agua | Discreto (coordenadas enteras) |
| **CartPole-v1** | Equilibrar un palo sobre un carrito | Continuo (4 valores reales) |

---

**Siguiente sección:** [9-Real-World](../9-Real-World/README.md) — Aplicaciones del mundo real
