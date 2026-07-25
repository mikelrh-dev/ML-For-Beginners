# Introduccion al aprendizaje por refuerzo

El aprendizaje por refuerzo (RL) se considera uno de los paradigmas basicos del aprendizaje automatico, junto con el aprendizaje supervisado y no supervisado. RL se trata sobre decisiones: tomar las decisiones correctas o al menos aprender de ellas.

Imagina que tienes un entorno simulado como la bolsa de valores. Que pasa si impones una regulacion determinada? Tiene un efecto positivo o negativo? Si algo negativo ocurre, necesitas tomar este _refuerzo negativo_, aprender de ello y cambiar el rumbo. Si el resultado es positivo, necesitas aprovechar ese _refuerzo positivo_.

![peter y el lobo](images/peter.png)

> Pedro y sus amigos necesitan escapar del lobo hambriento! Imagen de [Jen Looper](https://twitter.com/jenlooper)

## Tema regional: Pedro y el Lobo (Rusia)

[Pedro y el Lobo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) es un cuento musical escrito por el compositor ruso [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Es una historia sobre el joven pionero Pedro, que valientemente sale de su casa hacia el claro del bosque para perseguir al lobo. En esta seccion, entrenaremos algoritmos de aprendizaje automatico que ayudaran a Pedro:

- **Explorar** el area circundante y construir un mapa de navegacion optimal
- **Aprender** a usar una patineta y mantener el equilibrio para moverse mas rapido

[![Pedro y el Lobo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Haz clic en la imagen de arriba para escuchar Pedro y el Lobo de Prokofiev

## Aprendizaje por refuerzo

En las secciones anteriores, hemos visto dos ejemplos de problemas de aprendizaje automatico:

- **Supervisado**, donde tenemos conjuntos de datos que sugieren soluciones de ejemplo al problema que queremos resolver. La [Clasificacion](../4-Classification/README.md) y la [Regresion](../2-Regression/README.md) son tareas de aprendizaje supervisado.

- **No supervisado**, en el que no tenemos datos de entrenamiento etiquetados. El ejemplo principal de aprendizaje no supervisado es el [Clustering](../5-Clustering/README.md).

En esta seccion, te presentaremos un nuevo tipo de problema de aprendizaje que no requiere datos de entrenamiento etiquetados. Hay varios tipos de tales problemas:

- **[Aprendizaje semi-supervisado](https://wikipedia.org/wiki/Semi-supervised_learning)**, donde tenemos muchos datos sin etiquetas que se pueden usar para pre-entrenar el modelo.

- **[Aprendizaje por refuerzo](https://wikipedia.org/wiki/Reinforcement_learning)**, en el que un agente aprende como comportarse realizando experimentos en algun entorno simulado.

### Ejemplo - juego de computadora

Supongamos que quieres ensenar a una computadora a jugar un juego, como ajedrez o [Super Mario](https://wikipedia.org/wiki/Super_Mario). Para que la computadora pueda jugar un juego, necesitamos que prediga que movimiento hacer en cada uno de los estados del juego. Aunque esto puede parecer un problema de clasificacion, no lo es - porque no tenemos un conjunto de datos con estados y acciones correspondientes. Aunque podemos tener algunos datos como partidas de ajedrez existentes o grabaciones de jugadores jugando Super Mario, es probable que esos datos no cubran suficientemente un numero grande de estados posibles.

En lugar de buscar datos de juegos existentes, el **Aprendizaje por Refuerzo** (RL) se basa en la idea de *hacer que la computadora juegue* muchas veces y observe el resultado. Por lo tanto, para aplicar el Aprendizaje por Refuerzo, necesitamos dos cosas:

- **Un entorno** y **un simulador** que nos permitan jugar un juego muchas veces. Este simulador definiria todas las reglas del juego, asi como los posibles estados y acciones.

- **Una funcion de recompensa**, que nos diria que tan bien lo hicimos durante cada movimiento o juego.

La principal diferencia entre otros tipos de aprendizaje automatico y RL es que en RL normalmente no sabemos si ganamos o perdemos hasta que terminamos el juego. Por lo tanto, no podemos decir si un movimiento determinado es bueno o no - solo recibimos una recompensa al final del juego. Y nuestro objetivo es disenar algoritmos que nos permitan entrenar un modelo bajo condiciones inciertas. Aprenderemos sobre un algoritmo RL llamado **Q-Learning**.

## Lecciones

1. [Introduccion al aprendizaje por refuerzo y Q-Learning](1-QLearning/README-es.md)
2. [Usando un entorno de simulacion Gym](2-Gym/README-es.md)

## Creditos

"Introduccion al Aprendizaje por Refuerzo" fue escrito con ♥️ por [Dmitry Soshnikov](http://soshnikov.com)
