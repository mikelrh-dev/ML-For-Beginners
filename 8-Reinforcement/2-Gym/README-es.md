# Patinaje CartPole

El problema que hemos estado resolviendo en la leccion anterior puede parecer un problema de juguete, no realmente aplicable para escenarios de la vida real. No es el caso, porque muchos problemas del mundo real tambien comparten este escenario - incluyendo jugar Ajedrez o Go. Son similares porque tambien tenemos un tablero con reglas dadas y un **estado discreto**.

## [Quiz pre-clase](https://ff-quizzes.netlify.app/en/ml/)

## Introduccion

En esta leccion aplicaremos los mismos principios de Q-Learning a un problema con **estado continuo**, es decir, un estado que esta dado por uno o mas numeros reales. Trataremos con el siguiente problema:

> **Problema**: Si Pedro quiere escapar del lobo, necesita poder moverse mas rapido. Veremos como Pedro puede aprender a patinar, en particular, a mantener el equilibrio, usando Q-Learning.

![La gran escape!](images/escape.png)

> Pedro y sus amigos se ingenian para escapar del lobo! Imagen de [Jen Looper](https://twitter.com/jenlooper)

Usaremos una version simplificada del equilibrio conocida como el problema **CartPole**. En el mundo cartpole, tenemos un slider horizontal que puede moverse a la izquierda o derecha, y el objetivo es equilibrar un palo vertical encima del slider.

<img alt="un cartpole" src="images/cartpole.png" width="200"/>

## Prerequisitos

En esta leccion, usaremos una libreria llamada **OpenAI Gym** para simular diferentes **entornos**. Puedes ejecutar el codigo de esta leccion localmente (por ejemplo, desde Visual Studio Code), en cuyo caso la simulacion se abrira en una nueva ventana. Al ejecutar el codigo en linea, es posible que necesites hacer algunos ajustes al codigo, como se describe [aqui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

En la leccion anterior, las reglas del juego y el estado estaban dadas por la clase `Board` que definimos nosotros mismos. Aqui usaremos un **entorno de simulacion** especial, que simulare la fisica detras del palo equilibrado. Uno de los entornos de simulacion mas populares para entrenar algoritmos de aprendizaje por refuerzo se llama [Gym](https://gym.openai.com/), que es mantenido por [OpenAI](https://openai.com/). Usando este gym podemos crear diferentes **entornos** desde una simulacion de cartpole hasta juegos de Atari.

> **Nota**: Puedes ver otros entornos disponibles de OpenAI Gym [aqui](https://gym.openai.com/envs/#classic_control).

Primero, instalemos el gym e importemos las librerias necesarias (bloque de codigo 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Ejercicio - inicializar un entorno cartpole

Para trabajar con el problema del equilibrio cartpole, necesitamos inicializar el entorno correspondiente. Cada entorno esta asociado con un:

- **Espacio de observacion** que define la estructura de la informacion que recibimos del entorno. Para el problema cartpole, recibimos la posicion del palo, la velocidad y algunos otros valores.

- **Espacio de acciones** que define las acciones posibles. En nuestro caso el espacio de acciones es discreto, y consiste en dos acciones - **izquierda** y **derecha**. (bloque de codigo 2)

1. Para inicializar, escribe el siguiente codigo:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver como funciona el entorno, ejecutemos una simulacion corta de 100 pasos. En cada paso, proporcionamos una de las acciones a ejecutar - en esta simulacion simplemente elegimos aleatoriamente una accion de `action_space`.

1. Ejecuta el codigo a continuacion y mira a que conduce.

    ✅ Recuerda que es preferible ejecutar este codigo en la instalacion local de Python! (bloque de codigo 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Deberias ver algo similar a esta imagen:

    ![cartpole sin equilibrio](images/cartpole-nobalance.gif)

2. Durante la simulacion, necesitamos obtener observaciones para decidir como actuar. De hecho, la funcion step devuelve las observaciones actuales, una funcion de recompensa, y el flag done que indica si tiene sentido continuar la simulacion o no: (bloque de codigo 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Terminaras viendo algo como esto en la salida del notebook:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    El vector de observaciones que se devuelve en cada paso de la simulacion contiene los siguientes valores:
    - Posicion del carrito
    - Velocidad del carrito
    - Angulo del palo
    - Velocidad de rotacion del palo

3. Obten los valores minimo y maximo de esos numeros: (bloque de codigo 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Tambien puedes notar que el valor de recompensa en cada paso de la simulacion siempre es 1. Esto es porque nuestro objetivo es sobrevivir el mayor tiempo posible, es decir, mantener el palo en una posicion razonablemente vertical durante el mayor tiempo posible.

    ✅ De hecho, la simulacion CartPole se considera resuelta si logramos obtener una recompensa promedio de 195 en 100 pruebas consecutivas.

## Discretizacion del estado

En Q-Learning, necesitamos construir una Q-Table que defina que hacer en cada estado. Para poder hacer esto, necesitamos que el estado sea **discreto**, mas precisamente, debe contener un numero finito de valores discretos. Por lo tanto, necesitamos de alguna manera **discretizar** nuestras observaciones, mapeandolas a un conjunto finito de estados.

Hay algunas formas de hacer esto:

- **Dividir en contenedores**. Si conocemos el intervalo de un valor determinado, podemos dividir este intervalo en un numero de **contenedores**, y luego reemplazar el valor por el numero del contenedor al que pertenece. Esto se puede hacer usando el metodo numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). En este caso, sabremos exactamente el tamano del estado, porque dependera del numero de contenedores que seleccionemos para la digitalizacion.
  
  ✅ Podemos usar interpolacion lineal para llevar valores a un intervalo finito determinado (por ejemplo, de -20 a 20), y luego convertir numeros a enteros por redondeo. Esto nos da un poco menos de control sobre el tamano del estado, especialmente si no conocemos los rangos exactos de los valores de entrada. Por ejemplo, en nuestro caso 2 de 4 valores no tienen limites superiores/inferiores en sus valores, lo que puede resultar en un numero infinito de estados.

En nuestro ejemplo, usaremos el segundo enfoque. Como notaras mas adelante, a pesar de los limites superiores/inferiores indefinidos, esos valores raramente toman valores fuera de ciertos intervalos finitos, por lo que esos estados con valores extremos seran muy raros.

1. Aqui esta la funcion que tomara la observacion de nuestro modelo y producira una tupla de 4 valores enteros: (bloque de codigo 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

2. Exploremos otro metodo de discretizacion usando contenedores: (bloque de codigo 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Contenedores de ejemplo para el intervalo (-5,5) con 10 contenedores\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervalos de valores para cada parametro
    nbins = [20,20,10,10] # numero de contenedores para cada parametro
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

3. Ahora ejecutemos una simulacion corta y observemos esos valores discretos del entorno. Siéntete libre de probar tanto `discretize` como `discretize_bins` y ver si hay alguna diferencia.

    ✅ discretize_bins devuelve el numero del contenedor, que comienza en 0. Por lo tanto, para valores de la variable de entrada alrededor de 0 devuelve el numero del medio del intervalo (10). En discretize, no nos importaba el rango de los valores de salida, permitiendo que sean negativos, por lo que los valores del estado no estan desplazados, y 0 corresponde a 0. (bloque de codigo 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Descomenta la linea que comienza con env.render si quieres ver como se ejecuta el entorno. De lo contrario, puedes ejecutarlo en segundo plano, lo cual es mas rapido. Usaremos esta ejecucion "invisible" durante nuestro proceso de Q-Learning.

## La estructura de la Q-Table

En nuestra leccion anterior, el estado era un simple par de numeros del 0 al 8, y por lo tanto era conveniente representar la Q-Table con un tensor numpy con forma 8x8x2. Si usamos la discretizacion por contenedores, el tamano de nuestro vector de estado tambien es conocido, por lo que podemos usar el mismo enfoque y representar el estado por un array con forma 20x20x10x10x2 (aqui 2 es la dimension del espacio de acciones, y las primeras dimensiones corresponden al numero de contenedores que hemos seleccionado usar para cada uno de los parametros en el espacio de observacion).

Sin embargo, a veces las dimensiones exactas del espacio de observacion no son conocidas. En el caso de la funcion `discretize`, nunca podemos estar seguros de que nuestro estado se mantenga dentro de ciertos limites, porque algunos de los valores originales no estan limitados. Por lo tanto, usaremos un enfoque ligeramente diferente y representaremos la Q-Table con un diccionario.

1. Usa el par *(estado,accion)* como la clave del diccionario, y el valor corresponderia al valor de entrada de la Q-Table. (bloque de codigo 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aqui tambien definimos una funcion `qvalues()`, que devuelve una lista de valores de la Q-Table para un estado dado que corresponde a todas las acciones posibles. Si la entrada no esta presente en la Q-Table, devolveremos 0 como valor predeterminado.

## Comencemos Q-Learning

Ahora estamos listos para ensenar a Pedro a equilibrar!

1. Primero, establezcamos algunos hiperparametros: (bloque de codigo 10)

    ```python
    # hiperparametros
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aqui, `alpha` es la **tasa de aprendizaje** que define en que medida deberiamos ajustar los valores actuales de la Q-Table en cada paso. En la leccion anterior comenzamos con 1, y luego disminuimos `alpha` a valores mas bajos durante el entrenamiento. En este ejemplo lo mantendremos constante por simplicidad, y puedes experimentar ajustando los valores de `alpha` despues.

    `gamma` es el **factor de descuento** que muestra en que medida deberiamos priorizar la recompensa futura sobre la recompensa actual.

    `epsilon` es el **factor de exploracion/explotacion** que determina si deberiamos preferir la exploracion a la explotacion o viceversa. En nuestro algoritmo, en el `epsilon` por ciento de los casos seleccionaremos la siguiente accion segun los valores de la Q-Table, y en el numero restante de casos ejecutaremos una accion aleatoria. Esto nos permitira explorar areas del espacio de busqueda que nunca hemos visto antes.

    ✅ En terminos de equilibrio - elegir una accion aleatoria (exploracion) actuaria como un golpe aleatorio en la direccion incorrecta, y el palo tendria que aprender como recuperar el equilibrio de esos "errores".

### Mejorar el algoritmo

Tambien podemos hacer dos mejoras a nuestro algoritmo de la leccion anterior:

- **Calcular la recompensa acumulada promedio**, sobre un numero de simulaciones. Imprimiremos el progreso cada 5000 iteraciones, y promediaremos nuestra recompensa acumulada durante ese periodo de tiempo. Significa que si obtenemos mas de 195 puntos - podemos considerar el problema resuelto, con incluso mayor calidad de la requerida.
  
  - **Calcular el resultado acumulado promedio maximo**, `Qmax`, y almacenaremos la Q-Table correspondiente a ese resultado. Cuando ejecutes el entrenamiento notaras que a veces el resultado acumulado promedio comienza a caer, y queremos mantener los valores de la Q-Table que corresponden al mejor modelo observado durante el entrenamiento.

1. Recopila todas las recompensas acumuladas en cada simulacion en el vector `rewards` para graficarlas despues. (bloque de codigo 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == ejecutar la simulacion ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # explotacion - elegir la accion segun las probabilidades de la Q-Table
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploracion - elegir aleatoriamente la accion
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Imprimir resultados periodicamente y calcular recompensa promedio ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Lo que puedes notar de estos resultados:

- **Cerca de nuestro objetivo**. Estamos muy cerca de lograr el objetivo de obtener 195 recompensas acumuladas en mas de 100 ejecuciones consecutivas de la simulacion, o podemos haberlo logrado! Incluso si obtenemos numeros mas pequenos, aun no sabemos, porque promediamos sobre 5000 ejecuciones, y solo se requieren 100 ejecuciones en los criterios formales.
  
  - **La recompensa comienza a caer**. A veces la recompensa comienza a caer, lo que significa que podemos "danar" valores ya aprendidos en la Q-Table con los que hacen la situacion peor.

Esta observacion es mas claramente visible si graficamos el progreso del entrenamiento.

## Graficando el Progreso del Entrenamiento

Durante el entrenamiento, hemos recopilado el valor de la recompensa acumulada en cada una de las iteraciones en el vector `rewards`. Asi es como se ve cuando lo graficamos contra el numero de iteracion:

```python
plt.plot(rewards)
```

![progreso crudo](images/train_progress_raw.png)

De esta grafica, no es posible decir nada, porque debido a la naturaleza del proceso de entrenamiento estocastico, la duracion de las sesiones de entrenamiento varia mucho. Para tener mas sentido esta grafica, podemos calcular el **promedio movil** sobre una serie de experimentos, digamos 100. Esto se puede hacer convenientemente usando `np.convolve`: (bloque de codigo 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progreso del entrenamiento](images/train_progress_runav.png)

## Variando Hiperparametros

Para hacer el aprendizaje mas estable, tiene sentido ajustar algunos de nuestros hiperparametros durante el entrenamiento. En particular:

- **Para la tasa de aprendizaje**, `alpha`, podemos comenzar con valores cercanos a 1, y luego seguir disminuyendo el parametro. Con el tiempo, obtendremos buenos valores de probabilidad en la Q-Table, y por lo tanto deberiamos ajustarlos ligeramente, y no sobreescribirlos completamente con nuevos valores.

- **Incrementar epsilon**. Podemos querer incrementar `epsilon` lentamente, para explorar menos y explotar mas. Probablemente tenga sentido comenzar con un valor mas bajo de `epsilon`, y subir hasta casi 1.

> **Tarea 1**: Juega con los valores de los hiperparametros y mira si puedes lograr una recompensa acumulada mas alta. Estas llegando arriba de 195?

> **Tarea 2**: Para resolver formalmente el problema, necesitas obtener una recompensa promedio de 195 en 100 ejecuciones consecutivas. Mide eso durante el entrenamiento y asegurate de haber resuelto formalmente el problema!

## Viendo el resultado en accion

Seria interesante ver realmente como se comporta el modelo entrenado. Ejecutemos la simulacion y sigamos la misma estrategia de seleccion de acciones que durante el entrenamiento, muestreando segun la distribucion de probabilidad en la Q-Table: (bloque de codigo 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Deberias ver algo como esto:

![un cartpole equilibrado](images/cartpole-balance.gif)

---

## Desafio

> **Tarea 3**: Aqui, estabamos usando la copia final de la Q-Table, que puede no ser la mejor. Recuerda que hemos almacenado la Q-Table con mejor desempeno en la variable `Qbest`! Prueba el mismo ejemplo con la Q-Table de mejor desempeno copiando `Qbest` sobre `Q` y mira si notas la diferencia.

> **Tarea 4**: Aqui no estabamos seleccionando la mejor accion en cada paso, sino muestreando con la distribucion de probabilidad correspondiente. Tendria mas sentido siempre seleccionar la mejor accion, con el valor mas alto de la Q-Table? Esto se puede hacer usando la funcion `np.argmax` para encontrar el numero de accion correspondiente al valor mas alto de la Q-Table. Implementa esta estrategia y mira si mejora el equilibrio.

## [Quiz post-clase](https://ff-quizzes.netlify.app/en/ml/)

## Tarea
[Entrena un Mountain Car](assignment.md)

## Conclusion

Ahora hemos aprendido a entrenar agentes para lograr buenos resultados simplemente proporcionandoles una funcion de recompensa que define el estado deseado del juego, y dandoles la oportunidad de explorar intelligentemente el espacio de busqueda. Hemos aplicado exitosamente el algoritmo Q-Learning en los casos de entornos discretos y continuos, pero con acciones discretas.

Tambien es importante estudiar situaciones donde el estado de accion tambien es continuo, y cuando el espacio de observacion es mucho mas complejo, como la imagen de la pantalla de un juego de Atari. En esos problemas a menudo necesitamos usar tecnicas de aprendizaje automatico mas poderosas, como redes neuronales, para lograr buenos resultados. Esos temas mas avanzados son el sujeto de nuestro proximo curso de IA mas avanzado.
