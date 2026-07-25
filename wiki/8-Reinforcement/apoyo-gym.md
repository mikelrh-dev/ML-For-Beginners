# Apoyo — Lección 2: CartPole con Gym

## Resumen rápido

Aplicamos Q-Learning a un problema con **estado continuo** (CartPole) usando **OpenAI Gym**. La clave: discretizar las observaciones para poder usar una Q-Table.

---

## CartPole: qué es

Un carrito con un palo encima. El agente empuja izquierda (0) o derecha (1). Objetivo: mantener el palo vertical el mayor tiempo posible.

### Vector de observación

```python
obs = [posición, velocidad, ángulo, velocidad_angular]
```

| Componente | Rango típico | Unidad |
|------------|-------------|--------|
| Posición | [-2.4, 2.4] | Metros |
| Velocidad | [-∞, ∞] | m/s |
| Ángulo | [-0.21, 0.21] | Radianes |
| Vel. angular | [-∞, ∞] | rad/s |

---

## ¿Por qué discretizar?

Q-Learning necesita estados **discretos** (finitos). CartPole devuelve 4 números reales → hay que convertirlos a enteros.

### Método 1: División directa
```python
def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
```
Rápido, simple. No controlas el rango exacto.

### Método 2: Bins
```python
def create_bins(i, num):
    return np.arange(num+1) * (i[1]-i[0]) / num + i[0]

bins = [create_bins((-5,5),20), create_bins((-2,2),20),
        create_bins((-0.5,0.5),10), create_bins((-2,2),10)]

def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))
```
Controlas el rango. 20×20×10×10 = 40,000 estados posibles.

---

## Q-Table como diccionario

```python
Q = {}
actions = (0, 1)

def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]
```

- **Clave**: `(estado, acción)`
- **Valor por defecto**: 0 (estado nunca visitado)
- Ventaja: no necesitas saber el tamaño exacto del espacio de estados

---

## Hiperparámetros

```python
alpha = 0.3
gamma = 0.9
epsilon = 0.90
```

| Parámetro | Valor | Qué controla |
|-----------|-------|--------------|
| **Alpha** | 0.3 | Cuánto pesa la nueva información (fijo, no decae) |
| **Gamma** | 0.9 | Importancia de recompensas futuras (alto = visión lejana) |
| **Epsilon** | 0.90 | 90% explotación, 10% exploración |

### Diferencia con Lección 1

| Concepto | Lección 1 | Lección 2 |
|----------|-----------|-----------|
| Alpha | Variable (decae con `exp(-n/10e5)`) | Fijo (0.3) |
| Gamma | 0.5 | 0.9 |
| Exploración | Vía `probs()` (probabilístico) | Vía `epsilon` (explícito) |
| Q-Table | numpy array 8×8×4 | Diccionario `{(s,a): valor}` |

---

## Entrenamiento

```python
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
        Q[(s, a)] = (1-alpha)*Q.get((s,a),0) + alpha*(rew + gamma*max(qvalues(ns)))
```

### Mejoras sobre Lección 1

- **Qmax tracking**: guardamos la mejor Q-Table (`Qbest`)
- **Running average**: `np.convolve(x, np.ones(100)/100, mode='valid')`
- **Epsilon explícito**: control más preciso de exploración

---

## Criterio de solución

> CartPole se considera **resuelto** con **recompensa promedio ≥ 195** en 100 episodios consecutivos.

Recompensa máxima por episodio: 500 (env步数 límite).

---

## Preguntas frecuentes

**P: ¿Por qué no podemos usar el estado continuo directamente?**
R: Q-Learning se basa en una tabla finita. Con continuos tendrías infinitos estados → tabla infinita → imposible.

**P: ¿40,000 estados no son muchos?**
R: Sí, por eso necesitamos 100,000 epochs (vs 5,000 en Lección 1). La mayoría de los estados nunca se visitan gracias al diccionario.

**P: ¿Cuál método de discretización es mejor?**
R: Depende. División directa es más simple. Bins da más control sobre el rango. Ambos funcionan para CartPole.

**P: ¿Por qué gamma es más alto aquí (0.9 vs 0.5)?**
R: En CartPole, una mala decisión ahora puede causar la caída 10 pasos después. Gamma alto permite "conectar" esa acción temprana con el castigo tardío.

**P: ¿Qué pasa si epsilon = 1 siempre?**
R: Solo explotas lo conocido, nunca exploras. El agente se queda con lo primero que encontró (probablemente sub-óptimo).

---

## Resumen rápido

```
CartPole = problema de equilibrio con estado continuo
              ↓
Discretizamos las 4 observaciones a enteros
              ↓
Q-Table = diccionario {(estado, acción): valor}
              ↓
Alpha=0.3, Gamma=0.9, Epsilon=0.90
              ↓
100,000 epochs de entrenamiento
              ↓
Running average para ver progreso
Criterio: ≥ 195 de promedio en 100 episodios
```

---

**Volver al [índice de Reinforcement](README.md)**
