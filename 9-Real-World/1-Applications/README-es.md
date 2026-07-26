# Epílogo: Machine learning en el mundo real


![Resumen del machine learning en el mundo real en un sketchnote](../../sketchnotes/ml-realworld.png)
> Sketchnote de [Tomomi Imura](https://www.twitter.com/girlie_mac)

En este plan de estudios, ha aprendido muchas formas de preparar datos para el entrenamiento y crear modelos de machine learning. Ha construido una serie de modelos clásicos de regresión, clustering, clasificación, procesamiento de lenguaje natural y series temporales. ¡Felicidades! Ahora, quizás se pregunte para qué sirve todo esto... ¿cuáles son las aplicaciones reales de estos modelos?

Si bien la IA, que generalmente utiliza deep learning, ha generado gran interés en la industria, aún existen aplicaciones valiosas para los modelos clásicos de machine learning. ¡Incluso podría estar usando algunas de estas aplicaciones hoy! En esta lección, explorará cómo ocho industrias y dominios temáticos diferentes utilizan este tipo de modelos para hacer que sus aplicaciones sean más eficientes, confiables, inteligentes y valiosas para los usuarios.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanzas

El sector financiero ofrece muchas oportunidades para el machine learning. Muchos problemas en esta área se prestan para ser modelados y resueltos mediante ML.

### Detección de fraudes con tarjetas de crédito

Aprendimos sobre [k-means clustering](../../5-Clustering/2-K-Means/README.md) anteriormente en el curso, pero ¿cómo se puede usar para resolver problemas relacionados con el fraude en tarjetas de crédito?

K-means clustering resulta útil en una técnica de detección de fraude en tarjetas de crédito llamada **detección de anomalías** (outlier detection). Los valores atípicos, o desviaciones en las observaciones de un conjunto de datos, pueden indicarnos si una tarjeta de crédito se está usando de forma normal o si algo inusual está ocurriendo. Como se muestra en el artículo enlazado a continuación, puede ordenar los datos de tarjetas de crédito usando un algoritmo de k-means clustering y asignar cada transacción a un cluster según cuán atípica parezca ser. Luego, puede evaluar los clusters más riesgosos para transacciones fraudulentas versus legítimas.
[Referencia](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestión de patrimonios

En la gestión de patrimonios, una persona o empresa maneja inversiones en nombre de sus clientes. Su trabajo es mantener y hacer crecer el patrimonio a largo plazo, por lo que es esencial elegir inversiones que tengan un buen rendimiento.

Una forma de evaluar el rendimiento de una inversión particular es mediante la regresión estadística. La [regresión lineal](../../2-Regression/1-Tools/README.md) es una herramienta valiosa para comprender cómo se desempeña un fondo en relación con algún punto de referencia. También podemos deducir si los resultados de la regresión son estadísticamente significativos o cuánto afectarían las inversiones de un cliente. Incluso podría ampliar su análisis utilizando regresión múltiple, donde se pueden tener en cuenta factores de riesgo adicionales. Para ver un ejemplo de cómo funcionaría esto para un fondo específico, consulte el artículo a continuación sobre la evaluación del rendimiento de fondos mediante regresión.
[Referencia](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educación

El sector educativo también es un área muy interesante donde se puede aplicar el ML. Existen problemas interesantes por abordar, como detectar trampa en exámenes o ensayos, o gestionar el sesgo, intencional o no, en el proceso de corrección.

### Predicción del comportamiento estudiantil

[Coursera](https://coursera.com), un proveedor de cursos abiertos en línea, tiene un excelente blog técnico donde discuten muchas decisiones de ingeniería. En este caso de estudio, trazaron una línea de regresión para intentar explorar la correlación entre una calificación baja de NPS (Net Promoter Score) y la retención o abandono del curso.
[Referencia](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigación de sesgos

[Grammarly](https://grammarly.com), un asistente de escritura que verifica errores ortográficos y gramaticales, utiliza sofisticados [sistemas de procesamiento de lenguaje natural](../../6-NLP/README.md) en todos sus productos. Publicaron un interesante caso de estudio en su blog técnico sobre cómo abordaron el sesgo de género en machine learning, que aprendió en nuestra [lección introductoria sobre equidad](../../1-Introduction/3-fairness/README.md).
[Referencia](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Comercio minorista

El sector minorista sin duda puede beneficiarse del uso de ML, desde crear una mejor experiencia de cliente hasta gestionar el inventario de forma óptima.

### Personalización de la experiencia del cliente

En Wayfair, una empresa que vende artículos para el hogar como muebles, ayudar a los clientes a encontrar los productos adecuados para su gusto y necesidades es primordial. En este artículo, los ingenieros de la empresa describen cómo usan ML y NLP para "mostrar los resultados correctos a los clientes". En particular, su Query Intent Engine se ha construido para usar extracción de entidades, entrenamiento de clasificadores, extracción de activos y opiniones, y etiquetado de sentimientos en las reseñas de los clientes. Este es un caso de uso clásico de cómo funciona el NLP en el comercio minorista en línea.
[Referencia](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestión de inventario

Empresas innovadoras y ágiles como [StitchFix](https://stitchfix.com), un servicio de cajas que envía ropa a los consumidores, dependen en gran medida del ML para las recomendaciones y la gestión de inventario. Sus equipos de estilismo trabajan junto con sus equipos de merchandising; de hecho: "uno de nuestros científicos de datos experimentó con un algoritmo genético y lo aplicó a la indumentaria para predecir qué sería una prenda exitosa que no existe hoy. Llevamos eso al equipo de merchandising y ahora pueden usarlo como herramienta".
[Referencia](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Salud

El sector de la salud puede aprovechar el ML para optimizar tareas de investigación y también problemas logísticos como el reingreso de pacientes o la prevención de la propagación de enfermedades.

### Gestión de ensayos clínicos

La toxicidad en los ensayos clínicos es una gran preocupación para los fabricantes de medicamentos. ¿Cuánta toxicidad es tolerable? En este estudio, el análisis de varios métodos de ensayos clínicos llevó al desarrollo de un nuevo enfoque para predecir las probabilidades de los resultados de los ensayos clínicos. Específicamente, pudieron usar random forest para producir un [clasificador](../../4-Classification/README.md) capaz de distinguir entre grupos de fármacos.
[Referencia](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestión de reingresos hospitalarios

La atención hospitalaria es costosa, especialmente cuando los pacientes deben ser readmitidos. Este artículo analiza una empresa que usa ML para predecir la probabilidad de reingreso mediante algoritmos de [clustering](../../5-Clustering/README.md). Estos clusters ayudan a los analistas a "descubrir grupos de reingresos que podrían compartir una causa común".
[Referencia](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestión de enfermedades

La reciente pandemia ha puesto de relieve las formas en que el machine learning puede ayudar a detener la propagación de enfermedades. En este artículo, reconocerá el uso de ARIMA, curvas logísticas, regresión lineal y SARIMA. "Este trabajo es un intento de calcular la tasa de propagación de este virus y, por lo tanto, predecir las muertes, recuperaciones y casos confirmados, para que pueda ayudarnos a prepararnos mejor y sobrevivir".
[Referencia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecología y Tecnología Verde

La naturaleza y la ecología consisten en muchos sistemas sensibles donde la interacción entre animales y naturaleza cobra protagonismo. Es importante poder medir estos sistemas con precisión y actuar adecuadamente si ocurre algo, como un incendio forestal o una disminución en la población animal.

### Gestión forestal

Aprendió sobre [Reinforcement Learning](../../8-Reinforcement/README.md) en lecciones anteriores. Puede ser muy útil para predecir patrones en la naturaleza. En particular, se puede utilizar para rastrear problemas ecológicos como incendios forestales y la propagación de especies invasoras. En Canadá, un grupo de investigadores utilizó Reinforcement Learning para construir modelos de dinámica de incendios forestales a partir de imágenes satelitales. Usando un innovador "proceso de propagación espacial (SSP)", concibieron un incendio forestal como "el agente en cualquier celda del paisaje". "El conjunto de acciones que el fuego puede tomar desde una ubicación en cualquier momento incluye propagarse hacia el norte, sur, este, oeste o no propagarse".

Este enfoque invierte la configuración habitual de RL ya que la dinámica del correspondiente Proceso de Decisión de Markov (MDP) es una función conocida para la propagación inmediata de incendios forestales." Lea más sobre los algoritmos clásicos utilizados por este grupo en el enlace a continuación.
[Referencia](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Detección de movimiento de animales

Si bien el deep learning ha creado una revolución en el seguimiento visual de movimientos animales (puede construir su propio [rastreador de osos polares](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) aquí), el ML clásico todavía tiene un lugar en esta tarea.

Los sensores para rastrear movimientos de animales de granja e IoT hacen uso de este tipo de procesamiento visual, pero las técnicas de ML más básicas son útiles para preprocesar datos. Por ejemplo, en este artículo, se monitorearon y analizaron posturas de ovejas utilizando varios algoritmos clasificadores. Podría reconocer la curva ROC en la página 335.
[Referencia](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestión de energía

En nuestras lecciones sobre [pronóstico de series temporales](../../7-TimeSeries/README.md), mencionamos el concepto de parquímetros inteligentes para generar ingresos para una ciudad basándose en la comprensión de la oferta y la demanda. Este artículo analiza en detalle cómo la combinación de clustering, regresión y pronóstico de series temporales ayudó a predecir el uso futuro de energía en Irlanda, basándose en medidores inteligentes.
[Referencia](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Seguros

El sector de seguros es otro sector que utiliza ML para construir y optimizar modelos financieros y actuariales viables.

### Gestión de volatilidad

MetLife, un proveedor de seguros de vida, es transparente en la forma en que analiza y mitiga la volatilidad en sus modelos financieros. En este artículo notará visualizaciones de clasificación binaria y ordinal. También descubrirá visualizaciones de pronósticos.
[Referencia](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Arte, Cultura y Literatura

En las artes, por ejemplo en el periodismo, existen muchos problemas interesantes. La detección de noticias falsas es un problema enorme, ya que se ha demostrado que influye en la opinión de las personas e incluso derriba democracias. Los museos también pueden beneficiarse del uso de ML en todo, desde encontrar vínculos entre artefactos hasta la planificación de recursos.

### Detección de noticias falsas

Detectar noticias falsas se ha convertido en un juego del gato y el ratón en los medios de comunicación actuales. En este artículo, los investigadores sugieren que se puede probar un sistema que combine varias de las técnicas de ML que hemos estudiado y desplegar el mejor modelo: "Este sistema se basa en el procesamiento de lenguaje natural para extraer características de los datos y luego estas características se utilizan para entrenar clasificadores de machine learning como Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) y Logistic Regression (LR)".
[Referencia](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Este artículo muestra cómo la combinación de diferentes dominios de ML puede producir resultados interesantes que ayuden a evitar que las noticias falsas se propaguen y causen daños reales; en este caso, el impulso fue la propagación de rumores sobre tratamientos para COVID que incitaron a la violencia de masas.

### ML en museos

Los museos están en la cúspide de una revolución de la IA en la que catalogar y digitalizar colecciones y encontrar vínculos entre artefactos se está volviendo más fácil a medida que avanza la tecnología. Proyectos como [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) están ayudando a desbloquear los misterios de colecciones inaccesibles como los Archivos del Vaticano. Pero el aspecto comercial de los museos también se beneficia de los modelos de ML.

Por ejemplo, el Art Institute of Chicago construyó modelos para predecir qué interesa al público y cuándo asistirán a las exposiciones. El objetivo es crear experiencias de visitante individualizadas y optimizadas cada vez que el usuario visita el museo. "Durante el año fiscal 2017, el modelo predijo la asistencia y las admisiones con una precisión del 1 por ciento, dice Andrew Simnick, vicepresidente senior del Art Institute".
[Referencia](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentación de clientes

Las estrategias de marketing más efectivas se dirigen a los clientes de diferentes maneras según diversas agrupaciones. En este artículo, se discuten los usos de los algoritmos de Clustering para apoyar el marketing diferenciado. El marketing diferenciado ayuda a las empresas a mejorar el reconocimiento de marca, llegar a más clientes y generar más ingresos.
[Referencia](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Desafío

Identifique otro sector que se beneficie de algunas de las técnicas que aprendió en este plan de estudios y descubra cómo utiliza el ML.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

El equipo de ciencia de datos de Wayfair tiene varios videos interesantes sobre cómo usan el ML en su empresa. Vale la pena [echarles un vistazo](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos).

## Asignación

[Una búsqueda del tesoro de ML](assignment.md)
