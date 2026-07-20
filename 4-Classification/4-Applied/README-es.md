# Crea una aplicación web de recomendación de cocinas

En esta lección, construirás un modelo de clasificación utilizando algunas de las técnicas que has aprendido en lecciones anteriores y con el delicioso conjunto de datos de cocinas utilizado a lo largo de esta serie. Además, crearás una pequeña aplicación web para usar un modelo guardado, aprovechando el runtime web de Onnx.

Uno de los usos prácticos más útiles del aprendizaje automático es la construcción de sistemas de recomendación, ¡y hoy puedes dar el primer paso en esa dirección!

[![Presentando esta aplicación web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "ML Aplicado")

> 🎥 Haz clic en la imagen de arriba para ver un video: Jen Looper construye una aplicación web usando datos de cocinas clasificadas

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

En esta lección aprenderás:

- Cómo construir un modelo y guardarlo como un modelo Onnx
- Cómo usar Netron para inspeccionar el modelo
- Cómo usar tu modelo en una aplicación web para inferencia

## Construye tu modelo

Construir sistemas de ML aplicados es una parte importante para aprovechar estas tecnologías en tus sistemas empresariales. Puedes usar modelos dentro de tus aplicaciones web (y así usarlos en un contexto fuera de línea si es necesario) utilizando Onnx.

En una [lección anterior](../../3-Web-App/1-Web-App/README.md), construiste un modelo de Regresión sobre avistamientos de ovnis, lo "encurtiste" y lo usaste en una aplicación Flask. Si bien esta arquitectura es muy útil de conocer, es una aplicación Python de pila completa, y tus requisitos pueden incluir el uso de una aplicación JavaScript.

En esta lección, puedes construir un sistema básico basado en JavaScript para inferencia. Primero, sin embargo, necesitas entrenar un modelo y convertirlo para usarlo con Onnx.

## Ejercicio - entrenar un modelo de clasificación

Primero, entrena un modelo de clasificación utilizando el conjunto de datos de cocinas limpio que usamos.

1. Comienza importando las librerías útiles:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Necesitas '[skl2onnx](https://onnx.ai/sklearn-onnx/)' para ayudar a convertir tu modelo de Scikit-learn al formato Onnx.

1. Luego, trabaja con tus datos de la misma manera que lo hiciste en lecciones anteriores, leyendo un archivo CSV usando `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Elimina las dos primeras columnas innecesarias y guarda los datos restantes como 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Guarda las etiquetas como 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Inicia la rutina de entrenamiento

Usaremos la librería 'SVC' que tiene buena precisión.

1. Importa las librerías apropiadas de Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separa los conjuntos de entrenamiento y prueba:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Construye un modelo de clasificación SVC como lo hiciste en la lección anterior:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Ahora, prueba tu modelo llamando a `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Imprime un informe de clasificación para verificar la calidad del modelo:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Como vimos antes, la precisión es buena:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224

        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Convierte tu modelo a Onnx

Asegúrate de hacer la conversión con el número de Tensor adecuado. Este conjunto de datos tiene 380 ingredientes listados, por lo que debes notificar ese número en `FloatTensorType`:

1. Convierte usando un número de tensor de 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Crea el onx y guárdalo como un archivo **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Nota, puedes pasar [opciones](https://onnx.ai/sklearn-onnx/parameterized.html) en tu script de conversión. En este caso, pasamos 'nocl' como True y 'zipmap' como False. Dado que es un modelo de clasificación, tienes la opción de eliminar ZipMap que produce una lista de diccionarios (no es necesario). `nocl` se refiere a que la información de clase se incluye en el modelo. Reduce el tamaño de tu modelo estableciendo `nocl` en 'True'.

Ejecutar el notebook completo ahora construirá un modelo Onnx y lo guardará en esta carpeta.

## Visualiza tu modelo

Los modelos Onnx no son muy visibles en Visual Studio Code, pero hay un software gratuito muy bueno que muchos investigadores usan para visualizar el modelo y asegurarse de que esté correctamente construido. Descarga [Netron](https://github.com/lutzroeder/Netron) y abre tu archivo model.onnx. Puedes ver tu modelo simple visualizado, con sus 380 entradas y el clasificador listados:

![Visualización Netron](images/netron.png)

Netron es una herramienta útil para visualizar tus modelos.

Ahora estás listo para usar este modelo limpio en una aplicación web. Construyamos una aplicación que será útil cuando mires en tu refrigerador y trates de descubrir qué combinación de tus ingredientes sobrantes puedes usar para cocinar una cocina determinada, según lo determine tu modelo.

## Construye una aplicación web recomendadora

Puedes usar tu modelo directamente en una aplicación web. Esta arquitectura también te permite ejecutarlo localmente e incluso fuera de línea si es necesario. Comienza creando un archivo `index.html` en la misma carpeta donde guardaste tu archivo `model.onnx`.

1. En este archivo _index.html_, agrega el siguiente marcado:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Ahora, trabajando dentro de las etiquetas `body`, agrega un poco de marcado para mostrar una lista de casillas de verificación que reflejen algunos ingredientes:

    ```html
    <h1>Revisa tu refrigerador. ¿Qué puedes crear?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>manzana</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pera</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cereza</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenogreco</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>salsa de soja</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>comino</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">¿Qué tipo de cocina puedes preparar?</button>
            </div> 
    ```

    Observa que cada casilla de verificación tiene un valor. Esto refleja el índice donde se encuentra el ingrediente según el conjunto de datos. La manzana, por ejemplo, en esta lista alfabética, ocupa la quinta columna, por lo que su valor es '4' ya que comenzamos a contar desde 0. Puedes consultar la [hoja de cálculo de ingredientes](../data/ingredient_indexes.csv) para descubrir el índice de un ingrediente dado.

    Continuando tu trabajo en el archivo index.html, agrega un bloque de script donde se llama al modelo después del `</div>` final.

1. Primero, importa el [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime se utiliza para permitir ejecutar tus modelos Onnx en una amplia gama de plataformas de hardware, incluyendo optimizaciones y una API para usar.

1. Una vez que el Runtime está en su lugar, puedes llamarlo:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Por favor selecciona al menos un ingrediente.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('¡Hoy puedes disfrutar de cocina ' + results.label.data[0] + '!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

En este código, ocurren varias cosas:

1. Creaste un arreglo de 380 valores posibles (1 o 0) para ser establecidos y enviados al modelo para inferencia, dependiendo de si una casilla de ingrediente está marcada.
2. Creaste un arreglo de casillas de verificación y una forma de determinar si están marcadas en una función `init` que se llama cuando la aplicación se inicia. Cuando se marca una casilla, el arreglo `ingredients` se modifica para reflejar el ingrediente elegido.
3. Creaste una función `testCheckboxes` que verifica si alguna casilla está marcada.
4. Usas la función `startInference` cuando se presiona el botón y, si alguna casilla está marcada, inicias la inferencia.
5. La rutina de inferencia incluye:
   1. Configurar una carga asíncrona del modelo
   2. Crear una estructura Tensor para enviar al modelo
   3. Crear 'feeds' que reflejen la entrada `float_input` que creaste al entrenar tu modelo (puedes usar Netron para verificar ese nombre)
   4. Enviar estos 'feeds' al modelo y esperar una respuesta

## Prueba tu aplicación

Abre una sesión de terminal en Visual Studio Code en la carpeta donde se encuentra tu archivo index.html. Asegúrate de tener [http-server](https://www.npmjs.com/package/http-server) instalado globalmente y escribe `http-server` en el indicador. Se abrirá un localhost y podrás ver tu aplicación web. Comprueba qué cocina se recomienda según varios ingredientes:

![aplicación web de ingredientes](images/web-app.png)

¡Felicidades, has creado una aplicación web de 'recomendación' con algunos campos! ¡Tómate un tiempo para ampliar este sistema!

## 🚀Desafío

Tu aplicación web es muy mínima, así que continúa desarrollándola usando ingredientes y sus índices de los datos de [ingredient_indexes](../data/ingredient_indexes.csv). ¿Qué combinaciones de sabores funcionan para crear un plato nacional determinado?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Si bien esta lección solo tocó la utilidad de crear un sistema de recomendación para ingredientes alimenticios, esta área de aplicaciones de ML es muy rica en ejemplos. Lee más sobre cómo se construyen estos sistemas:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Asignación

[Construye un nuevo recomendador](assignment.md)
