# Lección 2: CartPole — Q-Learning con Estado Continuo

> **El mundo real no es una cuadrícula. Las posiciones, velocidades y ángulos son números reales. ¿Cómo aplicamos Q-Learning cuando el estado es continuo?**

---

## ¿Qué aprendemos aquí?

1. **El problema CartPole**: equilibrar un palo sobre un carrito
2. **OpenAI Gym**: biblioteca de entornos de simulación
3. **Estado continuo vs discreto**: por qué necesitamos discretizar
4. **Dos métodos de discretización**: división directa y bins
5. **Q-Table como diccionario**: cuando no sabemos el tamaño exacto del estado
6. **Hiperparámetros**: alpha, gamma, epsilon — roles y diferencias con Lección 1
7. **Entrenamiento**: 100,000 epochs, tracking de Qmax, running average
8. **Criterio de solución**: promedio ≥ 195 en 100 episodios consecutivos

---

## El problema CartPole

Un carrito se mueve horizontalmente. Sobre él hay un palo que debe mantenerse vertical. El agente puede empujar el carrito a la **izquierda** o a la **derecha**.

**Observaciones** (4 valores continuos):

| Variable | Rango | Descripción |
|----------|-------|-------------|
| Posición del carrito | ~[-4.8, 4.8] | Dónde está en el eje horizontal |
| Velocidad del carrito | [-∞, ∞] | Qué tan rápido se mueve |
| Ángulo del palo | ~[-0.42, 0.42] rad | Inclinación respecto a vertical |
| Velocidad angular | [-∞, ∞] | Qué tan rápido se inclina |

**Recompensa**: +1 por cada paso que el palo se mantenga vertical. El objetivo es **sobrevivir el mayor tiempo posible**.

---

## OpenAI Gym

```python
import gym

env = gym.make("CartPole-v1")
print(env.action_space)      # Discrete(2) → 0: izquierda, 1: derecha
print(env.observation_space) # Box(4,) → 4 valores continuos
```

### Acciones: izquierda (0) o derecha (1)

```python
actions = (0, 1)
```

### Loop básico de simulación

```python
env.reset()
done = False
while not done:
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample())
env.close()
```

El vector `obs` contiene: `[posición, velocidad, ángulo, velocidad_angular]`.

---

## Discretización del estado

Q-Learning necesita estados **discretos** (finitos). Las observaciones de CartPole son continuas → hay que discretizarlas.

### Método 1: División directa

```python
def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
```

Divide cada variable por un factor fijo y redondea a entero.

### Método 2: Bins (contenedores)

```python
def create_bins(i, num):
    return np.arange(num + 1) * (i[1] - i[0]) / num + i[0]

ints = [(-5, 5), (-2, 2), (-0.5, 0.5), (-2, 2)]
nbins = [20, 20, 10, 10]
bins = [create_bins(ints[i], nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))
```

Cada variable se mapea a un bucket. Tamaño total del espacio: 20 × 20 × 10 × 10 = 40,000 estados posibles.

---

## Q-Table como diccionario

En Lección 1 usábamos un numpy array (8×8×4) porque conocíamos las dimensiones exactas. Con discretización, el tamaño puede ser impredecible → usamos un **diccionario**:

```python
Q = {}
actions = (0, 1)

def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]
```

**Clave**: `(estado, acción)`. Si no existe, devuelve 0 por defecto. Esto permite manejar estados que nunca se visitaron.

---

## Hiperparámetros

```python
alpha = 0.3
gamma = 0.9
epsilon = 0.90
```

| Parámetro | Lección 1 | Lección 2 | Diferencia |
|-----------|-----------|-----------|------------|
| **Alpha (α)** | Variable (decae) | Fijo (0.3) | En CartPole el aprendizaje es más estable con alpha fijo |
| **Gamma (γ)** | 0.5 | 0.9 | CartPole necesita mirar más al futuro (el palo cae en varios pasos) |
| **Epsilon (ε)** | Implícito (vía probs) | 0.90 explotación | Control explícito de exploración vs explotación |

**Epsilon**: en el 90% de los casos elegimos según Q-Table (explotación). En el 10% restante, exploramos aleatoriamente.

```python
if random.random() < epsilon:
    # Explotación: elegir según Q-Table
    v = probs(np.array(qvalues(s)))
    a = random.choices(actions, weights=v)[0]
else:
    # Exploración: acción aleatoria
    a = np.random.randint(env.action_space.n)
```

---

## Entrenamiento (100,000 epochs)

```python
Qmax = 0
cum_rewards = []
rewards = []

for epoch in range(100000):
    obs = env.reset()
    done = False
    cum_reward = 0

    while not done:
        s = discretize(obs)
        if random.random() < epsilon:
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions, weights=v)[0]
        else:
            a = np.random.randint(env.action_space.n)

        obs, rew, done, info = env.step(a)
        cum_reward += rew
        ns = discretize(obs)
        Q[(s, a)] = (1 - alpha) * Q.get((s, a), 0) + alpha * (rew + gamma * max(qvalues(ns)))

    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)

    if epoch % 5000 == 0:
        avg = np.average(cum_rewards)
        print(f"{epoch}: avg={avg}, alpha={alpha}, epsilon={epsilon}")
        if avg > Qmax:
            Qmax = avg
            Qbest = Q
        cum_rewards = []
```

### Mejora respecto a Lección 1

| Mejora | Por qué |
|--------|---------|
| **Qmax tracking** | Guardamos la mejor Q-Table (la recompensa puede caer) |
| **Running average** | Suaviza el ruido del entrenamiento |
| **Epsilon explícito** | Control más fino de exploración |
| **Recompensa acumulada** | Medimos supervivencia, no pasos hasta meta |

---

## Evaluación con running average

La recompensa por episodio es muy ruidosa (0 a 500+). Para ver la tendencia real:

```python
def running_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode='valid')

plt.plot(running_average(rewards, 100))
```

### Criterio de solución formal

> El problema CartPole se considera **resuelto** si se obtiene una recompensa promedio ≥ 195 en 100 episodios consecutivos.

---

## Viendo el resultado

```python
obs = env.reset()
done = False
while not done:
    s = discretize(obs)
    env.render()
    v = probs(np.array(qvalues(s)))
    a = random.choices(actions, weights=v)[0]
    obs, _, done, _ = env.step(a)
env.close()
```

---

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| No discretizar el estado | Q-Learning no funciona con continuos |
| Discretizar con bins muy finos | Q-Table explota en tamaño (maldición de dimensionalidad) |
| Epsilon demasiado bajo | El agente nunca explora nuevas estrategias |
| Alpha = 1 todo el entrenamiento | Cada nuevo paso borra lo aprendido |
| No guardar Qbest | El mejor modelo se pierde cuando la recompensa cae |

---

## Notas técnicas

- **100,000 epochs** es ~10-20x más que en Lección 1. El espacio de estados es mucho más grande.
- La Q-Table como diccionario es más flexible pero más lenta que un numpy array.
- El método de discretización afecta directamente la calidad del aprendizaje.
- `Qbest` preserva la mejor política incluso si el entrenamiento se deteriora después.

---

## ¿Qué sigue?

Este es el final de la sección de Reinforcement Learning. La técnica de Q-Learning con discretización es la base para algoritmos más avanzados como Deep Q-Networks (DQN).

---

**Volver al [índice de Reinforcement](README.md)**
