# Modelos de clustering para machine learning

El clustering es una tarea de machine learning que busca encontrar objetos que se asemejan entre sí y agruparlos en grupos llamados clusters. Lo que diferencia al clustering de otros enfoques en machine learning es que las cosas ocurren automáticamente; de hecho, es justo decir que es lo opuesto al aprendizaje supervisado.

## Tema regional: modelos de clustering para el gusto musical de una audiencia nigeriana 🎧

La diversa audiencia de Nigeria tiene gustos musicales diversos. Usando datos extraídos de Spotify (inspirados en [este artículo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), analicemos algo de música popular en Nigeria. Este conjunto de datos incluye información sobre el puntaje de 'bailabilidad' de varias canciones, 'acousticness', volumen, 'speechiness', popularidad y energía. ¡Será interesante descubrir patrones en estos datos!

![Un tocadiscos](./images/turntable.jpg)

> Foto de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> en <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
En esta serie de lecciones, descubrirás nuevas formas de analizar datos usando técnicas de clustering. El clustering es particularmente útil cuando tu conjunto de datos carece de etiquetas. Si tiene etiquetas, entonces las técnicas de clasificación como las que aprendiste en lecciones anteriores podrían ser más útiles. Pero en casos donde buscas agrupar datos no etiquetados, el clustering es una excelente manera de descubrir patrones.

> Existen herramientas útiles de bajo código que pueden ayudarte a aprender sobre el trabajo con modelos de clustering. Prueba [Azure ML para esta tarea](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lecciones

1. [Introducción al clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Créditos

Estas lecciones fueron escritas con 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) con revisiones útiles de [Rishit Dagli](https://rishit_dagli) y [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

El conjunto de datos [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) fue obtenido de Kaggle, extraído de Spotify.

Ejemplos útiles de K-Means que ayudaron en la creación de esta lección incluyen esta [exploración de iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [cuaderno introductorio](https://www.kaggle.com/prashant111/k-means-clustering-with-python), y este [ejemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).
