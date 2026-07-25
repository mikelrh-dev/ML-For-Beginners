# Glosario de Reinforcement — Aprendizaje por Refuerzo

## Términos fundamentales

### Aprendizaje por Refuerzo (Reinforcement Learning — RL)
Paradigma de ML donde un **agente** aprende a tomar **acciones** en un **entorno** para maximizar una **recompensa acumulada**. No hay datos etiquetados — el agente aprende de la experiencia.

```
Supervisado:  dato → etiqueta (aprendes de ejemplos)
No supervisado: dato → patrón (encuentras estructura)
Refuerzo:     agente + entorno → recompensa (prueba y error)
```

### Agente (Agent)
Entidad que toma decisiones. En nuestros ejemplos, Pedro (Lección 1) y el controlador del carrito (Lección 2) son los agentes.

### Entorno (Environment)
El mundo donde el agente opera. Define las reglas, estados y recompensas.

| Lección | Entorno | Descripción |
|---------|---------|-------------|
| 1 | Peter & Wolf | Tablero 8×8 con obstáculos |
| 2 | CartPole-v1 | Simulación física de equilibrio |

### Estado (State)
Situación actual del agente dentro del entorno.

- **Estado discreto**: número finito de posibilidades (ej: coordenadas (x,y) en tablero 8×8 = 64 estados)
- **Estado continuo**: valores reales (ej: [posición, velocidad, ángulo, vel_angular] en CartPole)

Para aplicar Q-Learning a estados continuos, necesitamos **discretizarlos**.

### Acción (Action)
Movimiento o decisión que el agente puede tomar en un estado dado.

| Entorno | Acciones disponibles |
|---------|---------------------|
| Peter & Wolf | U, D, L, R (4 acciones discretas) |
| CartPole | Izquierda (0), Derecha (1) — 2 acciones discretas |

### Recompensa (Reward)
Valor numérico que el agente recibe después de ejecutar una acción. Es la **señal de retroalimentación**.

- **Recompensa inmediata**: la que se recibe justo después de la acción
- **Recompensa acumulada** (cumulative reward): suma de todas las recompensas en un episodio
- **Recompensa diferida**: la recompensa importante puede llegar muchos pasos después de la acción que la causó

```
CartPole: +1 por cada paso que el palo siga vertical
Peter:    +10 por manzana, -10 por lobo, -0.1 por paso
```

---

## Algoritmos y conceptos

### Política (Policy)
Estrategia que mapea **estados → acciones**. Determina qué hacer en cada situación.

```python
# Política aleatoria
def random_policy(m):
    return random.choice(list(actions))

# Política entrenada (Q-Learning)
def qpolicy(m):
    x, y = m.human
    v = probs(Q[x, y])
    a = random.choices(list(actions), weights=v)[0]
    return a
```

### Q-Table / Q-Valores
Estructura de datos que almacena la "bondad" de cada acción en cada estado. Cada celda contiene un valor Q que representa la **recompensa futura esperada** si tomamos esa acción desde ese estado.

```python
# Lección 1: numpy array
Q = np.ones((8, 8, 4)) * 0.25

# Lección 2: diccionario
Q = {}
Q[(state, action)] = valor
```

### Ecuación de Bellman (Bellman Equation)
Fórmula que actualiza los valores de la Q-Table. Es el corazón del Q-Learning.

```
Q(s,a) ← (1-α)·Q(s,a) + α·(r + γ·max_a' Q(s',a'))
```

| Componente | Significado |
|-----------|-------------|
| `Q(s,a)` | Valor actual de la acción a en el estado s |
| `α` (alpha) | Tasa de aprendizaje (cuánto incorporamos lo nuevo) |
| `r` | Recompensa inmediata |
| `γ` (gamma) | Factor de descuento |
| `max_a' Q(s',a')` | Mejor valor posible en el siguiente estado s' |

### Alpha (α) — Tasa de aprendizaje
Controla cuánto pesa la nueva información versus la información existente.

- **Alpha alto (≈1)**: aprendizaje rápido, pero puede "olvidar" lo aprendido. Bueno al principio.
- **Alpha bajo (≈0.1)**: aprendizaje lento y estable. Bueno al final.

```python
# En Lección 1: alpha decrece con el tiempo
alpha = np.exp(-n / 10e5)  # Empieza en ~1, termina en ~0

# En Lección 2: alpha fijo
alpha = 0.3  # Constante durante todo el entrenamiento
```

### Gamma (γ) — Factor de descuento
Determina cuánto valoramos las recompensas futuras versus la recompensa inmediata.

- **γ = 0**: solo importa la recompensa inmediata (miope)
- **γ = 1**: todas las recompensas futuras importan igual que la inmediata
- **γ = 0.5-0.9**: balance típico

| Lección | Gamma | Por qué |
|---------|-------|---------|
| 1 (Peter) | 0.5 | El objetivo (manzana) está cerca, menos necesario mirar lejos |
| 2 (CartPole) | 0.9 | Una mala acción ahora puede causar caída 10 pasos después |

### Epsilon (ε) — Exploración / Explotación
Controla el balance entre **explorar** (probar acciones nuevas) y **explotar** (usar lo aprendido).

```python
if random.random() < epsilon:
    # Explotar: elegir según Q-Table (lo que sabemos)
    a = mejor_accion_segun_Q(s)
else:
    # Explorar: acción aleatoria (probar algo nuevo)
    a = random_action()
```

- **ε alto (0.9+)**: más explotación, menos exploración
- **ε bajo**: más exploración, menos explotación

### Exploración vs Explotación (Exploration vs Exploitation)
Dilema fundamental de RL:

| | Exploración | Explotación |
|---|-------------|-------------|
| **Qué hace** | Prueba acciones nuevas | Usa la mejor acción conocida |
| **Riesgo** | Puede ser peor | Nunca descubre algo mejor |
| **Beneficio** | Descubre caminos óptimos | Usa el conocimiento actual |

### Epoch / Episodio (Episode)
Una ejecución completa desde el estado inicial hasta un estado terminal.

```
Inicio → [acción 1] → estado 1 → [acción 2] → ... → estado terminal
         └─────────── 1 episodio ──────────────────┘
```

| Lección | Epochs | Por qué tantos |
|---------|--------|----------------|
| 1 | 5,000 | Entorno pequeño (64 estados) |
| 2 | 100,000 | Estado continuo discretizado (~40,000 estados posibles) |

---

## Herramientas

### OpenAI Gym
Biblioteca de entornos de simulación para RL. Mantenida por OpenAI. Proporciona desde problemas simples (CartPole) hasta juegos de Atari.

```python
import gym
env = gym.make("CartPole-v1")
```

### Espacio de observación (Observation Space)
Define la estructura de la información que el entorno devuelve al agente.

```python
env.observation_space  # Box(4,) → 4 valores continuos
env.observation_space.low   # [-4.8..., -inf, -0.42..., -inf]
env.observation_space.high  # [4.8...,  inf, 0.42...,  inf]
```

### Espacio de acciones (Action Space)
Define las acciones posibles que el agente puede ejecutar.

```python
env.action_space  # Discrete(2) → 0 o 1
```

### Discretización
Proceso de convertir valores continuos en discretos para poder usar Q-Learning.

```python
# División directa
def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))

# Bins
def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))
```

---

## Hiperparámetros vs Parámetros

| | Parámetros | Hiperparámetros |
|---|------------|-----------------|
| **Qué son** | Valores que el modelo aprende | Valores que nosotros configuramos |
| **Ejemplos** | Valores Q en Q-Table | Alpha, Gamma, Epsilon |
| **Cómo se definen** | Aprendidos durante entrenamiento | Elegidos antes de entrenar |
| **Optimización** | Automática (Bellman update) | Manual (prueba y error) |

---

## Conceptos de evaluación

### Running Average (Promedio móvil)
Suaviza una serie ruidosa promediando ventanas consecutivas.

```python
def running_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode='valid')
```

### Recompensa acumulada (Cumulative Reward)
Suma total de recompensas obtenidas en un episodio.

### Qmax tracking
Guardar la Q-Table que produjo la mejor recompensa promedio hasta el momento. Previene perder el mejor modelo cuando el entrenamiento se deteriora.

---

---

**Volver al [índice de Reinforcement](README.md)**
