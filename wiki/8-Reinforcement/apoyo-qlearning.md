# Apoyo — Lección 1: Q-Learning e Introducción a RL

## ¿Qué es Reinforcement Learning?

Un **agente** aprende a base de **prueba y error**. Recibe **recompensas** por sus acciones y aprende qué secuencia de pasos maximiza la recompensa total.

```
Analogía: entrenar a un perro
  - Se sienta → recibe galleta (+1)
  - No se sienta → no recibe nada (0)
  - Con repetición, aprende que sentarse = galleta
```

---

## El entorno: tablero 8x8

| Celda | Recompensa | Efecto |
|-------|------------|--------|
| Suelo | -0.1 | Puede caminar |
| Manzana | +10 | Gana el juego |
| Lobo | -10 | Muere |
| Agua | -10 | Muere |
| Árbol/Hierba | -0.1 | Seguro, igual que suelo |

Las acciones disponibles: **U** (arriba), **D** (abajo), **L** (izquierda), **R** (derecha).

---

## Código clave

### Crear el entorno
```python
from rlboard import *
m = Board(8, 8)
m.randomize(seed=13)
```

### Función de recompensa
```python
move_reward = -0.1   # castigo por cada paso
goal_reward = 10     # premio por manzana
end_reward = -10     # castigo por morir
```

### Inicializar Q-Table
```python
Q = np.ones((8, 8, 4), dtype=np.float) * 0.25  # 0.25 = 1/4 acciones
```

### Ecuación de Bellman (simplificada)
```
Nuevo Q = (1 - α) * Q_viejo + α * (recompensa + γ * mejor_Q_futuro)
```

---

## Hiperparámetros

| Parámetro | Valor | Qué hace |
|-----------|-------|----------|
| **Alpha (α)** | `exp(-n / 10e5)` — empieza en 1, decrece | Tasa de aprendizaje. Controla cuánto pesa la nueva información |
| **Gamma (γ)** | 0.5 | Factor de descuento. 0 = solo importa el ahora, 1 = todo importa |
| **Epochs** | 5000 | Número de episodios de entrenamiento |

### Alpha decay
```python
alpha = np.exp(-n / 10e5)
```
Al principio alpha ≈ 1 (aprende rápido). Al final alpha ≈ 0 (se estabiliza).

---

## Resultados esperados

| Estrategia | Pasos promedio | Mejora |
|------------|---------------|--------|
| Random Walk | ~30-40 | — |
| Q-Learning (entrenado) | ~3-6 | ~7x mejor |

---

## Exploración vs Explotación

| Estrategia | Ventaja | Riesgo |
|------------|---------|--------|
| Explotar (mejor acción conocida) | Usa lo aprendido | Nunca descubre caminos mejores |
| Explorar (acción aleatoria) | Descubre nuevas rutas | Puede morir más seguido |

Q-Learning usa `probs()` para balancear:

```python
def probs(v, eps=1e-4):
    v = v - v.min() + eps
    v = v / v.sum()
    return v
```

---

## Preguntas frecuentes

**P: ¿Por qué la Q-Table empieza con 0.25?**
R: Es 1/4 (4 acciones). Así todas tienen la misma probabilidad inicial = random walk.

**P: ¿Qué pasa si alpha = 1 siempre?**
R: El modelo olvida todo lo aprendido en cada paso. Nunca se estabiliza.

**P: ¿Por qué gamma = 0.5 y no 0.9?**
R: En este entorno, las recompensas futuras no están muy lejanas. Gamma más bajo hace al agente más "miope" pero más estable.

**P: ¿Qué significa que la longitud del camino aumente durante el entrenamiento?**
R: Es normal. El agente está explorando más territorios, lo que incluye meterse en peligros. Baja cuando el conocimiento madura.

**P: ¿Por qué la política estricta a veces se cuelga?**
R: Porque dos estados pueden tener valores Q que se apuntan mutuamente → bucle infinito. La política suave (probabilística) evita esto.

---

## Resumen rápido

```
RL = agente + entorno + acciones + recompensas
         ↓
Q-Table guarda "qué tan buena es cada acción en cada estado"
         ↓
Ecuación de Bellman actualiza la Q-Table
         ↓
Alpha controla velocidad de aprendizaje
Gamma controla miopía / visión a futuro
         ↓
Resultado: de 35 pasos aleatorios → 5 pasos inteligentes
```

---

**Siguiente paso**: [Lección 2: CartPole con Gym](lesson-2-gym.md) — Estado continuo y OpenAI Gym.

**Volver al [índice de Reinforcement](README.md)**
