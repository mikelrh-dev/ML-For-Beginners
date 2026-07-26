# Epílogo: Depuración de modelos en machine learning usando componentes del panel Responsible AI

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

El machine learning impacta nuestra vida cotidiana. La IA está llegando a algunos de los sistemas más importantes que nos afectan como individuos y como sociedad, desde la salud, las finanzas, la educación y el empleo. Por ejemplo, los sistemas y modelos participan en tareas diarias de toma de decisiones, como diagnósticos médicos o detección de fraudes. En consecuencia, los avances en IA junto con la adopción acelerada están siendo recibidos con expectativas sociales en evolución y una creciente regulación en respuesta. Constantemente vemos áreas donde los sistemas de IA siguen sin cumplir las expectativas; exponen nuevos desafíos; y los gobiernos están comenzando a regular las soluciones de IA. Por lo tanto, es importante que estos modelos sean analizados para proporcionar resultados justos, confiables, inclusivos, transparentes y responsables para todos.

En este plan de estudios, veremos herramientas prácticas que se pueden usar para evaluar si un modelo tiene problemas de IA responsable. Las técnicas tradicionales de depuración de machine learning tienden a basarse en cálculos cuantitativos como la precisión agregada o la pérdida de error promedio. Imagine lo que puede suceder cuando los datos que está utilizando para construir estos modelos carecen de ciertos datos demográficos, como raza, género, opinión política, religión, o representan de manera desproporcionada dichos datos demográficos. ¿Qué sucede cuando la salida del modelo se interpreta para favorecer a algún grupo demográfico? Esto puede introducir una sobrerrepresentación o subrepresentación de estos grupos de características sensibles, lo que resulta en problemas de equidad, inclusividad o confiabilidad del modelo. Otro factor es que los modelos de machine learning se consideran cajas negras, lo que dificulta entender y explicar qué impulsa la predicción de un modelo. Todos estos son desafíos que enfrentan los científicos de datos y desarrolladores de IA cuando no tienen herramientas adecuadas para depurar y evaluar la equidad o confiabilidad de un modelo.

En esta lección, aprenderá sobre la depuración de sus modelos utilizando:

- **Error Analysis**: identificar dónde en su distribución de datos el modelo tiene altas tasas de error.
- **Model Overview**: realizar análisis comparativos entre diferentes cohortes de datos para descubrir disparidades en las métricas de rendimiento de su modelo.
- **Data Analysis**: investigar dónde podría haber sobrerrepresentación o subrepresentación de sus datos que pueda sesgar su modelo para favorecer un grupo demográfico sobre otro.
- **Feature Importance**: comprender qué características están impulsando las predicciones de su modelo a nivel global o local.

## Prerrequisito

Como prerrequisito, revise [Responsible AI tools for developers](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif sobre herramientas de IA Responsable](./images/rai-overview.gif)

## Error Analysis

Las métricas de rendimiento de modelos tradicionales utilizadas para medir la precisión son principalmente cálculos basados en predicciones correctas versus incorrectas. Por ejemplo, determinar que un modelo es preciso el 89% del tiempo con una pérdida de error de 0.001 puede considerarse un buen rendimiento. Los errores a menudo no se distribuyen uniformemente en su conjunto de datos subyacente. Puede obtener una puntuación de precisión del modelo del 89% pero descubrir que hay diferentes regiones de sus datos para las cuales el modelo está fallando el 42% del tiempo. La consecuencia de estos patrones de fallo con ciertos grupos de datos puede llevar a problemas de equidad o confiabilidad. Es esencial comprender las áreas donde el modelo se está desempeñando bien o no. Las regiones de datos donde hay una alta cantidad de imprecisiones en su modelo podrían resultar ser un grupo demográfico de datos importante.

![Analizar y depurar errores del modelo](./images/ea-error-distribution.png)

El componente Error Analysis en el panel RAI ilustra cómo el fallo del modelo se distribuye entre varias cohortes con una visualización de árbol. Esto es útil para identificar características o áreas donde hay una alta tasa de error con su conjunto de datos. Al ver de dónde provienen la mayoría de las imprecisiones del modelo, puede comenzar a investigar la causa raíz. También puede crear cohortes de datos para realizar análisis. Estas cohortes de datos ayudan en el proceso de depuración para determinar por qué el rendimiento del modelo es bueno en una cohorte, pero erróneo en otra.

![Error Analysis](./images/ea-error-cohort.png)

Los indicadores visuales en el mapa de árbol ayudan a localizar las áreas problemáticas más rápidamente. Por ejemplo, cuanto más oscuro es el tono de color rojo que tiene un nodo del árbol, mayor es la tasa de error.

El mapa de calor es otra funcionalidad de visualización que los usuarios pueden usar para investigar la tasa de error utilizando una o dos características para encontrar un contribuyente a los errores del modelo en un conjunto de datos completo o cohortes.

![Mapa de calor de Error Analysis](./images/ea-heatmap.png)

Use error analysis cuando necesite:

- Obtener una comprensión profunda de cómo se distribuyen los fallos del modelo en un conjunto de datos y en varias dimensiones de entrada y características.
- Desglosar las métricas de rendimiento agregadas para descubrir automáticamente cohortes erróneas que informen sus pasos de mitigación específicos.

## Model Overview

Evaluar el rendimiento de un modelo de machine learning requiere obtener una comprensión holística de su comportamiento. Esto se puede lograr revisando más de una métrica, como tasa de error, precisión, exhaustividad (recall), o MAE (Mean Absolute Error) para encontrar disparidades entre las métricas de rendimiento. Una métrica de rendimiento puede verse excelente, pero pueden exponerse imprecisiones en otra métrica. Además, comparar las métricas en busca de disparidades en todo el conjunto de datos o cohortes ayuda a arrojar luz sobre dónde el modelo se está desempeñando bien o no. Esto es especialmente importante para ver el rendimiento del modelo entre características sensibles e insensibles (por ejemplo, raza, género o edad del paciente) para descubrir una posible falta de equidad que el modelo pueda tener. Por ejemplo, descubrir que el modelo es más erróneo en una cohorte que tiene características sensibles puede revelar una posible falta de equidad que el modelo pueda tener.

El componente Model Overview del panel RAI ayuda no solo a analizar las métricas de rendimiento de la representación de datos en una cohorte, sino que brinda a los usuarios la capacidad de comparar el comportamiento del modelo entre diferentes cohortes.

![Cohortes de datos - model overview en panel RAI](./images/model-overview-dataset-cohortes.png)

La funcionalidad de análisis basado en características del componente permite a los usuarios reducir subgrupos de datos dentro de una característica particular para identificar anomalías a nivel granular. Por ejemplo, el panel tiene inteligencia incorporada para generar automáticamente cohortes para una característica seleccionada por el usuario (p. ej., *"time_in_hospital < 3"* o *"time_in_hospital >= 7"*). Esto permite a un usuario aislar una característica particular de un grupo de datos más grande para ver si es un factor clave de los resultados erróneos del modelo.

![Cohortes de características - model overview en panel RAI](./images/model-overview-feature-cohortes.png)

El componente Model Overview admite dos clases de métricas de disparidad:

**Disparidad en el rendimiento del modelo**: estos conjuntos de métricas calculan la disparidad (diferencia) en los valores de la métrica de rendimiento seleccionada entre subgrupos de datos. Aquí hay algunos ejemplos:

- Disparidad en la tasa de precisión
- Disparidad en la tasa de error
- Disparidad en precisión (precision)
- Disparidad en exhaustividad (recall)
- Disparidad en el error absoluto medio (MAE)

**Disparidad en la tasa de selección**: esta métrica contiene la diferencia en la tasa de selección (predicción favorable) entre subgrupos. Un ejemplo de esto es la disparidad en las tasas de aprobación de préstamos. La tasa de selección significa la fracción de puntos de datos en cada clase clasificados como 1 (en clasificación binaria) o la distribución de valores de predicción (en regresión).

## Data Analysis

> "Si torturas los datos el tiempo suficiente, confesarán cualquier cosa" - Ronald Coase

Esta afirmación suena extrema, pero es cierto que los datos pueden ser manipulados para apoyar cualquier conclusión. Dicha manipulación puede ocurrir a veces sin intención. Como humanos, todos tenemos sesgos, y a menudo es difícil saber conscientemente cuándo se está introduciendo sesgo en los datos. Garantizar la equidad en la IA y el machine learning sigue siendo un desafío complejo.

Los datos son un gran punto ciego para las métricas tradicionales de rendimiento de modelos. Puede tener puntuaciones de alta precisión, pero esto no siempre refleja el sesgo de datos subyacente que podría estar en su conjunto de datos. Por ejemplo, si un conjunto de datos de empleados tiene un 27% de mujeres en puestos ejecutivos en una empresa y un 73% de hombres en el mismo nivel, un modelo de IA de publicidad de empleo entrenado con estos datos podría dirigirse principalmente a una audiencia masculina para puestos de alto nivel. Tener este desequilibrio en los datos sesgó la predicción del modelo para favorecer a un género. Esto revela un problema de equidad donde hay un sesgo de género en el modelo de IA.

El componente Data Analysis en el panel RAI ayuda a identificar áreas donde hay sobrerrepresentación y subrepresentación en el conjunto de datos. Ayuda a los usuarios a diagnosticar la causa raíz de errores y problemas de equidad introducidos por desequilibrios de datos o falta de representación de un grupo de datos particular. Esto brinda a los usuarios la capacidad de visualizar conjuntos de datos basados en resultados predichos y reales, grupos de error y características específicas. A veces, descubrir un grupo de datos subrepresentado también puede revelar que el modelo no está aprendiendo bien, de ahí las altas imprecisiones. Tener un modelo que tiene sesgo de datos no es solo un problema de equidad, sino que muestra que el modelo no es inclusivo ni confiable.

![Componente Data Analysis en el panel RAI](./images/dataanalysis-cover.png)

Use data analysis cuando necesite:

- Explorar las estadísticas de su conjunto de datos seleccionando diferentes filtros para segmentar sus datos en diferentes dimensiones (también conocidas como cohortes).
- Comprender la distribución de su conjunto de datos entre diferentes cohortes y grupos de características.
- Determinar si sus hallazgos relacionados con equidad, análisis de errores y causalidad (derivados de otros componentes del panel) son resultado de la distribución de su conjunto de datos.
- Decidir en qué áreas recopilar más datos para mitigar errores que provienen de problemas de representación, ruido en etiquetas, ruido en características, sesgo en etiquetas y factores similares.

## Model Interpretability

Los modelos de machine learning tienden a ser cajas negras. Comprender qué características clave de los datos impulsan la predicción de un modelo puede ser un desafío. Es importante proporcionar transparencia sobre por qué un modelo hace una predicción determinada. Por ejemplo, si un sistema de IA predice que un paciente diabético tiene riesgo de ser readmitido en un hospital en menos de 30 días, debería poder proporcionar datos de respaldo que llevaron a su predicción. Tener indicadores de datos de respaldo aporta transparencia para ayudar a los médicos u hospitales a tomar decisiones bien informadas. Además, poder explicar por qué un modelo hizo una predicción para un paciente individual permite la rendición de cuentas ante las regulaciones de salud. Cuando utiliza modelos de machine learning de maneras que afectan la vida de las personas, es crucial entender y explicar qué influye en el comportamiento de un modelo. La explicabilidad e interpretabilidad del modelo ayuda a responder preguntas en escenarios como:

- Depuración del modelo: ¿Por qué mi modelo cometió este error? ¿Cómo puedo mejorar mi modelo?
- Colaboración humano-IA: ¿Cómo puedo entender y confiar en las decisiones del modelo?
- Cumplimiento normativo: ¿Mi modelo cumple con los requisitos legales?

El componente Feature Importance del panel RAI le ayuda a depurar y obtener una comprensión integral de cómo un modelo hace predicciones. También es una herramienta útil para profesionales de machine learning y tomadores de decisiones para explicar y mostrar evidencia de las características que influyen en el comportamiento de un modelo para el cumplimiento normativo. A continuación, los usuarios pueden explorar explicaciones tanto globales como locales para validar qué características impulsan la predicción de un modelo. Las explicaciones globales enumeran las principales características que afectaron la predicción general de un modelo. Las explicaciones locales muestran qué características llevaron a la predicción de un modelo para un caso individual. La capacidad de evaluar explicaciones locales también es útil para depurar o auditar un caso específico para comprender e interpretar mejor por qué un modelo hizo una predicción precisa o imprecisa.

![Componente Feature Importance del panel RAI](./images/9-feature-importance.png)

- Explicaciones globales: Por ejemplo, ¿qué características afectan el comportamiento general de un modelo de readmisión hospitalaria por diabetes?
- Explicaciones locales: Por ejemplo, ¿por qué se predijo que un paciente diabético mayor de 60 años con hospitalizaciones previas sería readmitido o no readmitido en un hospital dentro de 30 días?

En el proceso de depuración al examinar el rendimiento de un modelo en diferentes cohortes, Feature Importance muestra qué nivel de impacto tiene una característica en las cohortes. Ayuda a revelar anomalías al comparar el nivel de influencia que tiene la característica en impulsar las predicciones erróneas del modelo. El componente Feature Importance puede mostrar qué valores en una característica influyeron positiva o negativamente en el resultado del modelo. Por ejemplo, si un modelo hizo una predicción inexacta, el componente le brinda la capacidad de profundizar y señalar qué características o valores de características impulsaron la predicción. Este nivel de detalle ayuda no solo en la depuración, sino que proporciona transparencia y rendición de cuentas en situaciones de auditoría. Finalmente, el componente puede ayudarle a identificar problemas de equidad. A modo de ejemplo, si una característica sensible como etnia o género es altamente influyente en impulsar la predicción de un modelo, esto podría ser una señal de sesgo racial o de género en el modelo.

![Importancia de características](./images/9-features-influence.png)

Use interpretability cuando necesite:

- Determinar qué tan confiables son las predicciones de su sistema de IA comprendiendo qué características son más importantes para las predicciones.
- Abordar la depuración de su modelo comprendiéndolo primero e identificando si el modelo está utilizando características saludables o meras correlaciones falsas.
- Descubrir fuentes potenciales de falta de equidad al comprender si el modelo basa sus predicciones en características sensibles o en características altamente correlacionadas con ellas.
- Generar confianza del usuario en las decisiones de su modelo generando explicaciones locales para ilustrar sus resultados.
- Completar una auditoría regulatoria de un sistema de IA para validar modelos y monitorear el impacto de las decisiones del modelo en los seres humanos.

## Conclusión

Todos los componentes del panel RAI son herramientas prácticas para ayudarle a construir modelos de machine learning que sean menos dañinos y más confiables para la sociedad. Mejora la prevención de amenazas a los derechos humanos; la discriminación o exclusión de ciertos grupos de oportunidades de vida; y el riesgo de lesiones físicas o psicológicas. También ayuda a generar confianza en las decisiones de su modelo al generar explicaciones locales para ilustrar sus resultados. Algunos de los daños potenciales se pueden clasificar como:

- **Asignación (Allocation)**, si un género o etnia, por ejemplo, es favorecido sobre otro.
- **Calidad del servicio**. Si entrena los datos para un escenario específico pero la realidad es mucho más compleja, esto lleva a un servicio de bajo rendimiento.
- **Estereotipos (Stereotyping)**. Asociar un grupo determinado con atributos preasignados.
- **Denigración (Denigration)**. Criticar y etiquetar injustamente algo o alguien.
- **Sobrerrepresentación o subrepresentación**. La idea de que cierto grupo no se ve en una profesión determinada, y cualquier servicio o función que siga promoviendo eso contribuye al daño.

### Panel Azure RAI

El [panel Azure RAI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) está construido sobre herramientas de código abierto desarrolladas por las principales instituciones académicas y organizaciones, incluyendo Microsoft, y son fundamentales para que los científicos de datos y desarrolladores de IA comprendan mejor el comportamiento de los modelos, descubran y mitiguen problemas no deseados en los modelos de IA.

- Aprenda a usar los diferentes componentes consultando la [documentación](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) del panel RAI.

- Consulte algunos [cuadernos de ejemplo](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) del panel RAI para depurar más escenarios de IA responsable en Azure Machine Learning.

---
## 🚀 Desafío

Para evitar que se introduzcan sesgos estadísticos o de datos en primer lugar, deberíamos:

- tener diversidad de antecedentes y perspectivas entre las personas que trabajan en los sistemas
- invertir en conjuntos de datos que reflejen la diversidad de nuestra sociedad
- desarrollar mejores métodos para detectar y corregir el sesgo cuando ocurre

Piense en escenarios de la vida real donde la falta de equidad sea evidente en la construcción y el uso de modelos. ¿Qué más deberíamos considerar?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

En esta lección, ha aprendido algunas de las herramientas prácticas para incorporar IA responsable en el machine learning.

Vea este taller para profundizar en los temas:

- Responsible AI Dashboard: One-stop shop for operationalizing RAI in practice por Besmira Nushi y Mehrnoosh Sameki

[![Responsible AI Dashboard: One-stop shop for operationalizing RAI in practice](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: One-stop shop for operationalizing RAI in practice")

> 🎥 Haga clic en la imagen de arriba para ver el video: Responsible AI Dashboard: One-stop shop for operationalizing RAI in practice por Besmira Nushi y Mehrnoosh Sameki

Consulte los siguientes materiales para aprender más sobre IA responsable y cómo construir modelos más confiables:

- Herramientas del panel RAI de Microsoft para depurar modelos de ML: [Responsible AI tools resources](https://aka.ms/rai-dashboard)

- Explore el toolkit de IA Responsable: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Centro de recursos de IA Responsable de Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupo de investigación FATE de Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Asignación

[Explore RAI dashboard](assignment.md)
