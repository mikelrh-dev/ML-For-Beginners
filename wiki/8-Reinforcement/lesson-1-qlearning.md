# Lección 1: Q-Learning — Introducción al Aprendizaje por Refuerzo

> **El agente no sabe qué hacer. Prueba, falla, aprende, repite. Así nace la inteligencia.**

---

## ¿Qué aprendemos aquí?

1. **Qué es Reinforcement Learning** y en qué se diferencia de otros paradigmas
2. **El entorno de Pedro y el Lobo**: tablero 8x8 con obstáculos, manzanas y peligros
3. **Acciones y Política**: cómo el agente decide moverse
4. **Random Walk**: línea base para medir mejora
5. **Función de Recompensa**: cómo definir el objetivo del agente
6. **Q-Learning**: Q-Table, Ecuación de Bellman, entrenamiento
7. **Exploración vs Explotación**: el equilibrio fundamental de RL
8. **Verificar la política entrenada**: de ~35 pasos a ~3-6 pasos

---

## El entorno: Pedro y el Lobo

El mundo de Pedro es un tablero cuadrado de 8x8 con 5 tipos de celda:

| Celda | Color | Significado |
|-------|-------|-------------|
| Suelo | Café claro | Pedro puede caminar |
| Agua | Azul | No se puede caminar |
| Árbol/Hierba | Verde | Lugar seguro para descansar |
| Manzana | Rojo | Objetivo: +10 recompensa |
| Lobo | Gris oscuro | Peligro: -10 recompensa |

```python
from rlboard import *

width, height = 8, 8
m = Board(width, height)
m.randomize(seed=13)
m.plot()
```

---

## Acciones y Política

En cada posición, Pedro puede elegir entre 4 acciones:

```python
actions = { "U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0) }
action_idx = { a: i for i, a in enumerate(actions.keys()) }
```

La **política** es la estrategia que define qué acción tomar en cada estado. La política más simple: **random walk** (camino aleatorio).

```python
def random_policy(m):
    return random.choice(list(actions))
```

### Resultado del random walk (~35 pasos promedio)

```python
def walk(m, policy, start_position=None):
    n = 0
    if start_position:
        m.human = start_position
    else:
        m.random_start()
    while True:
        if m.at() == Board.Cell.apple:
            return n
        if m.at() in [Board.Cell.wolf, Board.Cell.water]:
            return -1
        while True:
            a = actions[policy(m)]
            new_pos = m.move_pos(m.human, a)
            if m.is_valid(new_pos) and m.at(new_pos) != Board.Cell.water:
                m.move(a)
                break
        n += 1

def print_statistics(policy):
    s, w, n = 0, 0, 0
    for _ in range(100):
        z = walk(m, policy)
        if z < 0:
            w += 1
        else:
            s += z
            n += 1
    print(f"Longitud promedio = {s/n}, comido por lobo: {w} veces")

print_statistics(random_policy)
```

**Resultado esperado**: ~30-40 pasos promedio, con varias muertes por lobo/agua. La distancia real a la manzana más cercana es de ~5-6 pasos — el agente es muy ineficiente.

---

## Función de Recompensa

El objetivo se define con una función que asigna un puntaje a cada estado:

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m, pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x == Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x == Board.Cell.apple:
        return goal_reward
    return move_reward
```

| Situación | Recompensa | Por qué |
|-----------|------------|---------|
| Moverse por el tablero | -0.1 | Pequeño castigo por cada paso (queremos eficiencia) |
| Llegar a la manzana | +10 | Gran recompensa por cumplir el objetivo |
| Caer en agua o lobo | -10 | Castigo severo por morir |

**Clave**: la recompensa sustancial solo llega al final. El algoritmo debe recordar qué pasos llevaron a esa recompensa positiva.

---

## Q-Learning

### Q-Table

Registra la "bondad" de cada acción en cada estado. Es un tensor de forma `ancho x alto x acciones`:

```python
Q = np.ones((width, height, len(actions)), dtype=np.float) * 1.0 / len(actions)
```

Inicializamos con **0.25** (1/4) para todos los estados — todo igual de probable, como el random walk.

### Ecuación de Bellman

La fórmula que actualiza los valores de la Q-Table:

```
Q(s,a) ← (1-α)Q(s,a) + α(r + γ·max_a' Q(s',a'))
```

| Símbolo | Significado |
|---------|-------------|
| `Q(s,a)` | Valor actual de la acción `a` en el estado `s` |
| `α` (alpha) | Tasa de aprendizaje — cuánto incorporamos lo nuevo |
| `r` | Recompensa inmediata recibida |
| `γ` (gamma) | Factor de descuento — importancia de recompensas futuras |
| `max_a' Q(s',a')` | Mejor valor posible en el siguiente estado |

### Algoritmo de aprendizaje

```python
for epoch in range(5000):
    m.random_start()
    n = 0
    cum_reward = 0
    while True:
        x, y = m.human
        v = probs(Q[x, y])
        a = random.choices(list(actions), weights=v)[0]
        dpos = actions[a]
        m.move(dpos, check_correctness=False)
        r = reward(m)
        cum_reward += r
        if r == end_reward or cum_reward < -1000:
            break
        alpha = np.exp(-n / 10e5)
        gamma = 0.5
        ai = action_idx[a]
        Q[x, y, ai] = (1 - alpha) * Q[x, y, ai] + alpha * (r + gamma * Q[x + dpos[0], y + dpos[1]].max())
        n += 1
```

**Decaimiento de alpha**: `α = exp(-n / 10e5)` — al principio aprende rápido, luego se estabiliza.

### Exploración vs Explotación

La función `probs()` convierte los valores Q en probabilidades:

```python
def probs(v, eps=1e-4):
    v = v - v.min() + eps
    v = v / v.sum()
    return v
```

Esto permite elegir acciones **proporcionalmente** a su valor Q: a veces explotamos (elegir la mejor), a veces exploramos (probar algo nuevo).

---

## Verificando la política entrenada

### Política estricta (siempre la mejor acción)

```python
def qpolicy_strict(m):
    x, y = m.human
    v = probs(Q[x, y])
    a = list(actions)[np.argmax(v)]
    return a
```

Puede "colgarse" en bucles infinitos si dos estados se señalan mutuamente.

### Política suave (probabilística)

```python
def qpolicy(m):
    x, y = m.human
    v = probs(Q[x, y])
    a = random.choices(list(actions), weights=v)[0]
    return a

print_statistics(qpolicy)
```

**Resultado**: ~3-6 pasos promedio. De 35 a 5 pasos — una mejora de ~7x.

---

## Comportamiento del aprendizaje

| Fase | Longitud del camino | Explicación |
|------|-------------------|-------------|
| Inicio (~0 epochs) | ~35 pasos | Random walk, agente no sabe nada |
| Temprano (~500) | Aumenta | Explora más, encuentra peligros con más frecuencia |
| Intermedio (~2000) | Disminuye | Empieza a recordar caminos a la manzana |
| Final (~5000) | ~3-6 pasos | Aprendizaje maduro, camino casi óptimo |

**Ojo**: la longitud puede aumentar abruptamente — es la naturaleza estocástica del aprendizaje. Por eso es importante reducir alpha gradualmente.

---

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| Alpha demasiado alto | El modelo "olvida" lo aprendido constantemente |
| Gamma demasiado bajo | El agente solo busca recompensa inmediata |
| No explorar lo suficiente | El agente nunca descubre mejores caminos |
| Q-Table mal inicializada | Sesgo hacia ciertas acciones desde el inicio |
| No usar decaimiento de alpha | El modelo nunca se estabiliza |

---

## ¿Qué sigue?

[**Lección 2: CartPole con Gym**](lesson-2-gym.md) → Pasar de estado discreto a continuo con OpenAI Gym.

---

## Notas técnicas

- El módulo `rlboard.py` contiene todo el entorno. No necesitas modificarlo.
- La Q-Table se inicializa con valores iguales (0.25) = política neutral.
- Los **hiperparámetros** (alpha, gamma) se eligen empíricamente. No hay fórmula mágica.
- 5000 epochs es suficiente para este entorno pequeño. Entornos más complejos necesitan más.

---

**Volver al [índice de Reinforcement](README.md)**
