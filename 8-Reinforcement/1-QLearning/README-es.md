# Introduccion al Aprendizaje por Refuerzo y Q-Learning

![Resumen del aprendizaje por refuerzo en machine learning en un sketchnote](../../../sketchnotes/ml-reinforcement.png)
> Sketchnote de [Tomomi Imura](https://www.twitter.com/girlie_mac)

El aprendizaje por refuerzo involucra tres conceptos importantes: el agente, algunos estados y un conjunto de acciones por estado. Al ejecutar una accion en un estado especifico, el agente recibe una recompensa. Imagina de nuevo el juego de Super Mario. Tu eres Mario, estas en un nivel del juego, al lado del borde de un acantilado. Arriba de ti hay una moneda. Tu siendo Mario, en un nivel del juego, en una posicion especifica ... ese es tu estado. Moverte un paso a la derecha (una accion) te llevaria al borde, y eso te daria una puntuacion numerica baja. Sin embargo, presionar el boton de salto te permitiria ganar un punto y seguirias vivo. Ese es un resultado positivo y eso deberia recompensarte con una puntuacion numerica positiva.

Usando aprendizaje por refuerzo y un simulador (el juego), puedes aprender a jugar para maximizar la recompensa que es seguir vivo y obtener la mayor cantidad de puntos posible.

[![Introduccion al Aprendizaje por Refuerzo](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> �V Haz clic en la imagen de arriba para escuchar a Dmitry hablar sobre Aprendizaje por Refuerzo

## [Quiz pre-clase](https://ff-quizzes.netlify.app/en/ml/)

## Prerequisitos y Configuracion

En esta leccion, estaremos experimentando con algo de codigo en Python. Deberias poder ejecutar el codigo del Jupyter Notebook de esta leccion, ya sea en tu computadora o en la nube.

Puedes abrir [el notebook de la leccion](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) y seguir esta leccion para construir.

> **Nota:** Si abres este codigo desde la nube, tambien necesitas descargar el archivo [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que se usa en el codigo del notebook. Agregalo al mismo directorio que el notebook.

## Introduccion

En esta leccion, exploraremos el mundo de **[Pedro y el Lobo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirado en un cuento musical del compositor ruso [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Usaremos **Aprendizaje por Refuerzo** para permitir que Pedro explore su entorno, recolecte manzanas deliciosas y evite encontrarse con el lobo.

**El Aprendizaje por Refuerzo** (RL) es una tecnica de aprendizaje que nos permite aprender un comportamiento optimo de un **agente** en algun **entorno** ejecutando muchos experimentos. Un agente en este entorno deberia tener algun **objetivo**, definido por una **funcion de recompensa**.

## El entorno

Para simplificar, consideremos que el mundo de Pedro es un tablero cuadrado de tamano `ancho` x `altura`, asi:

![Entorno de Pedro](images/environment.png)

Cada celda en este tablero puede ser:

* **suelo**, donde Pedro y otras criaturas pueden caminar.
* **agua**, donde obviamente no puedes caminar.
* un **arbol** o **hierba**, un lugar donde puedes descansar.
* una **manzana**, que representa algo que Pedro estaria contento de encontrar para alimentarse.
* un **lobo**, que es peligroso y debe ser evitado.

Hay un modulo Python separado, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que contiene el codigo para trabajar con este entorno. Como este codigo no es importante para entender nuestros conceptos, importaremos el modulo y lo usaremos para crear el tablero de ejemplo (bloque de codigo 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Este codigo deberia imprimir una imagen del entorno similar a la de arriba.

## Acciones y politica

En nuestro ejemplo, el objetivo de Pedro seria encontrar una manzana, evitando al lobo y otros obstaculos. Para hacer esto, basicamente puede caminar hasta encontrar una manzana.

Por lo tanto, en cualquier posicion, puede elegir entre una de las siguientes acciones: arriba, abajo, izquierda y derecha.

Definiremos esas acciones como un diccionario, y las mapearemos a pares de cambios de coordenadas correspondientes. Por ejemplo, moverse a la derecha (`R`) corresponderia a un par `(1,0)`. (bloque de codigo 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Para resumir, la estrategia y el objetivo de este escenario son los siguientes:

- **La estrategia**, de nuestro agente (Pedro) esta definida por una llamada **politica**. Una politica es una funcion que devuelve la accion en cualquier estado dado. En nuestro caso, el estado del problema esta representado por el tablero, incluyendo la posicion actual del jugador.

- **El objetivo**, del aprendizaje por refuerzo es eventualmente aprender una buena politica que nos permita resolver el problema de manera eficiente. Sin embargo, como linea base, consideremos la politica mas simple llamada **camino aleatorio**.

## Camino aleatorio

Primero resolvamos nuestro problema implementando una estrategia de camino aleatorio. Con el camino aleatorio, elegiremos aleatoriamente la siguiente accion de las acciones permitidas hasta que lleguemos a la manzana (bloque de codigo 3).

1. Implementa el camino aleatorio con el siguiente codigo:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # numero de pasos
        # establecer posicion inicial
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # exito!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # comido por el lobo o ahogado
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # hacer el movimiento real
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    La llamada a `walk` deberia devolver la longitud del camino correspondiente, que puede variar de una ejecucion a otra.

2. Ejecuta el experimento del camino varias veces (digamos, 100) e imprime las estadisticas resultantes (bloque de codigo 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Longitud promedio del camino = {s/n}, comido por el lobo: {w} veces")
    
    print_statistics(random_policy)
    ```

    Nota que la longitud promedio del camino es de alrededor de 30-40 pasos, que es bastante, dado el hecho de que la distancia promedio a la manzana mas cercana es de alrededor de 5-6 pasos.

    Tambien puedes ver como se ve el movimiento de Pedro durante el camino aleatorio:

    ![Camino Aleatorio de Pedro](images/random_walk.gif)

## Funcion de recompensa

Para hacer nuestra politica mas inteligente, necesitamos entender que movimientos son "mejores" que otros. Para hacer esto, necesitamos definir nuestro objetivo.

El objetivo puede definirse en terminos de una **funcion de recompensa**, que devolvera algum valor de puntuacion para cada estado. Cuanto mayor sea el numero, mejor es la funcion de recompensa. (bloque de codigo 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Una cosa interesante sobre las funciones de recompensa es que en la mayoria de los casos, *solo recibimos una recompensa sustancial al final del juego*. Esto significa que nuestro algoritmo debe de alguna manera recordar los "buenos" pasos que llevaron a una recompensa positiva al final, e incrementar su importancia. De manera similar, todos los movimientos que lleven a malos resultados deben ser desalentados.

## Q-Learning

Un algoritmo que discutiremos aqui se llama **Q-Learning**. En este algoritmo, la politica esta definida por una funcion (o una estructura de datos) llamada **Q-Table**. Registra la "bondad" de cada una de las acciones en un estado determinado.

Se llama Q-Table porque a menudo es conveniente representarla como una tabla, o un array multidimensional. Dado que nuestro tablero tiene dimensiones `ancho` x `altura`, podemos representar la Q-Table usando un array numpy con forma `ancho` x `altura` x `len(acciones)`: (bloque de codigo 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Notemos que inicializamos todos los valores de la Q-Table con un valor igual, en nuestro caso - 0.25. Esto corresponde a la politica de "camino aleatorio", porque todos los movimientos en cada estado son igualmente buenos. Podemos pasar la Q-Table a la funcion `plot` para visualizar la tabla en el tablero: `m.plot(Q)`.

![Entorno de Pedro](images/env_init.png)

En el centro de cada celda hay una "flecha" que indica la direccion preferida de movimiento. Dado que todas las direcciones son iguales, se muestra un punto.

Ahora necesitamos ejecutar la simulacion, explorar nuestro entorno y aprender una mejor distribucion de los valores de la Q-Table, que nos permitira encontrar el camino a la manzana mucho mas rapido.

## Esencia de Q-Learning: Ecuacion de Bellman

Una vez que comencemos a movernos, cada accion tendra una recompensa correspondiente, es decir, teoricamente podemos seleccionar la siguiente accion basada en la recompensa inmediata mas alta. Sin embargo, en la mayoria de los estados, el movimiento no lograra nuestro objetivo de alcanzar la manzana, y por lo tanto no podemos decidir inmediatamente cual direccion es mejor.

> Recuerda que no es el resultado inmediato lo que importa, sino el resultado final, que obtendremos al final de la simulacion.

Para tener en cuenta esta recompensa retrasada, necesitamos usar los principios de **[programacion dinamica](https://en.wikipedia.org/wiki/Dynamic_programming)**, que nos permiten pensar en nuestro problema recursivamente.

Supongamos que ahora estamos en el estado *s*, y queremos movernos al siguiente estado *s'*. Al hacer esto, recibiremos la recompensa inmediata *r(s,a)*, definida por la funcion de recompensa, mas alguna recompensa futura. Si suponemos que nuestra Q-Table refleja correctamente la "atractividad" de cada accion, entonces en el estado *s'* elegiremos una accion *a* que corresponda al valor maximo de *Q(s',a')*. Por lo tanto, la mejor recompensa futura posible que podriamos obtener en el estado *s* se definira como `max`<sub>a'</sub>*Q(s',a')* (el maximo aqui se calcula sobre todas las acciones posibles *a'* en el estado *s'*).

Esto nos da la **formula de Bellman** para calcular el valor de la Q-Table en el estado *s*, dada la accion *a*:

<img src="images/bellman-equation.png"/>

Aqui gamma es el llamado **factor de descuento** que determina en que medida deberias preferir la recompensa actual sobre la recompensa futura y viceversa.

## Algoritmo de Aprendizaje

Dada la ecuacion anterior, ahora podemos escribir el pseudocodigo para nuestro algoritmo de aprendizaje:

* Inicializar la Q-Table Q con numeros iguales para todos los estados y acciones
* Establecer la tasa de aprendizaje alpha ← 1
* Repetir la simulacion muchas veces
   1. Comenzar en una posicion aleatoria
   1. Repetir
        1. Seleccionar una accion *a* en el estado *s*
        2. Ejecutar la accion moviendose a un nuevo estado *s'*
        3. Si encontramos una condicion de fin de juego, o la recompensa total es muy pequena - salir de la simulacion
        4. Calcular la recompensa *r* en el nuevo estado
        5. Actualizar la funcion Q segun la ecuacion de Bellman: *Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. Actualizar la recompensa total y disminuir alpha.

## Explotar vs. explorar

En el algoritmo anterior, no especificamos como exactamente deberiamos elegir una accion en el paso 2.1. Si elegimos la accion aleatoriamente, exploraremos aleatoriamente el entorno, y es bastante probable que muramos frecuentemente, asi como exploremos areas a las que normalmente no iriamos. Un enfoque alternativo seria **explotar** los valores de la Q-Table que ya conocemos, y por lo tanto elegir la mejor accion (con mayor valor de Q-Table) en el estado *s*. Sin embargo, esto nos impediria explorar otros estados, y es probable que no encontremos la solucion optima.

Por lo tanto, el mejor enfoque es encontrar un equilibrio entre exploracion y explotacion. Esto se puede hacer eligiendo la accion en el estado *s* con probabilidades proporcionales a los valores en la Q-Table. Al principio, cuando los valores de la Q-Table son todos iguales, corresponderia a una seleccion aleatoria, pero a medida que aprendamos mas sobre nuestro entorno, seria mas probable que sigamos la ruta optima mientras permitimos que el agente elija el camino inexplorado de vez en cuando.

## Implementacion en Python

Ahora estamos listos para implementar el algoritmo de aprendizaje. Antes de hacerlo, tambien necesitamos alguna funcion que convierta numeros arbitrarios en la Q-Table en un vector de probabilidades para las acciones correspondientes.

1. Crea una funcion `probs()`:

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    Agregamos un poco de `eps` al vector original para evitar division por cero en el caso inicial, cuando todos los componentes del vector son identicos.

Ejecuta el algoritmo de aprendizaje a traves de 5000 experimentos, tambien llamados **epochs**: (bloque de codigo 8)
```python
    for epoch in range(5000):
    
        # Seleccionar punto inicial
        m.random_start()
        
        # Comenzar a viajar
        n=0
        cum_reward = 0
        while True:
            x,y = m.human
            v = probs(Q[x,y])
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.move(dpos,check_correctness=False) # permitimos que el jugador se mueva fuera del tablero, lo que termina el episodio
            r = reward(m)
            cum_reward += r
            if r==end_reward or cum_reward < -1000:
                lpath.append(n)
                break
            alpha = np.exp(-n / 10e5)
            gamma = 0.5
            ai = action_idx[a]
            Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
            n+=1
```

Despues de ejecutar este algoritmo, la Q-Table deberia actualizarse con valores que definan la atraccion de diferentes acciones en cada paso. Podemos intentar visualizar la Q-Table dibujando un vector en cada celda que apunte en la direccion deseada de movimiento. Para simplificar, dibujamos un pequeno circulo en lugar de una punta de flecha.

<img src="images/learned.png"/>

## Verificando la politica

Dado que la Q-Table enumera la "atractividad" de cada accion en cada estado, es bastante facil usarla para definir la navegacion eficiente en nuestro mundo. En el caso mas simple, podemos seleccionar la accion correspondiente al valor mas alto de la Q-Table: (bloque de codigo 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Si pruebas el codigo anterior varias veces, puedes notar que a veces se "cuelga", y necesitas presionar el boton STOP en el notebook para interrumpirlo. Esto sucede porque puede haber situaciones donde dos estados "se senalan" mutuamente en terminos del valor Q optimo, en cuyo caso el agente termina moviendose entre esos estados indefinidamente.

## Desafio

> **Tarea 1:** Modifica la funcion `walk` para limitar la longitud maxima del camino a un numero determinado de pasos (digamos, 100), y observa como el codigo anterior devuelve este valor de vez en cuando.

> **Tarea 2:** Modifica la funcion `walk` para que no vuelva a lugares donde ya estado antes. Esto evitara que `walk` entre en bucles, sin embargo, el agente aun puede terminar "atrapado" en una ubicacion de la que no puede escapar.

## Navegacion

Una mejor politica de navegacion seria la que usamos durante el entrenamiento, que combina explotacion y exploracion. En esta politica, seleccionaremos cada accion con cierta probabilidad, proporcional a los valores en la Q-Table. Esta estrategia aun puede resultar en que el agente vuelva a una posicion que ya ha explorado, pero, como puedes ver en el codigo a continuacion, resulta en un camino promedio muy corto a la ubicacion deseada (recuerda que `print_statistics` ejecuta la simulacion 100 veces): (bloque de codigo 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Despues de ejecutar este codigo, deberias obtener una longitud promedio del camino mucho menor que antes, en el rango de 3-6.

## Investigando el proceso de aprendizaje

Como hemos mencionado, el proceso de aprendizaje es un equilibrio entre la exploracion y la explotacion del conocimiento adquirido sobre la estructura del espacio del problema. Hemos visto que los resultados del aprendizaje (la capacidad de ayudar a un agente a encontrar un camino corto al objetivo) han mejorado, pero tambien es interesante observar como se comporta la longitud promedio del camino durante el proceso de aprendizaje:

<img src="images/lpathlen1.png"/>

Los aprendizajes se pueden resumir como:

- **La longitud promedio del camino aumenta**. Lo que vemos aqui es que al principio, la longitud promedio del camino aumenta. Esto probablemente se debe al hecho de que cuando no sabemos nada sobre el entorno, es probable que nos atrapen en malos estados, agua o lobo. A medida que aprendemos mas y comenzamos a usar este conocimiento, podemos explorar el entorno por mas tiempo, pero aun no sabemos donde estan las manzanas muy bien.

- **La longitud del camino disminuye a medida que aprendemos mas**. Una vez que aprendemos lo suficiente, se vuelve mas facil para el agente lograr el objetivo, y la longitud del camino comienza a disminuir. Sin embargo, aun estamos abiertos a la exploracion, por lo que a menudo nos desviamos del mejor camino y exploramos nuevas opciones, haciendo el camino mas largo que el optimo.

- **La longitud aumenta abruptamente**. Lo que tambien observamos en esta grafica es que en algun punto, la longitud aumento abruptamente. Esto indica la naturaleza estocastica del proceso, y que podemos en algun punto "danar" los coeficientes de la Q-Table sobreescribiendolos con nuevos valores. Esto idealmente deberia minimizarse disminuyendo la tasa de aprendizaje (por ejemplo, hacia el final del entrenamiento, solo ajustamos los valores de la Q-Table en un valor pequeno).

En general, es importante recordar que el exito y la calidad del proceso de aprendizaje dependen significativamente de parametros, como la tasa de aprendizaje, el decaimiento de la tasa de aprendizaje y el factor de descuento. Estos a menudo se llaman **hiperparametros**, para distinguirlos de los **parametros**, que optimizamos durante el entrenamiento (por ejemplo, los coeficientes de la Q-Table). El proceso de encontrar los mejores valores de hiperparametros se llama **optimizacion de hiperparametros**, y merece un tema separado.

## [Quiz post-clase](https://ff-quizzes.netlify.app/en/ml/)

## Tarea
[Un Mundo Mas Realista](assignment.md)
