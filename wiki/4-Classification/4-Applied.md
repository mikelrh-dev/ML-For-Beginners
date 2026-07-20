# 4. App de Recomendación de Cocinas 🍳

> "El mejor modelo es el que la gente puede usar."

## ¿Qué aprendemos aquí?

Juntamos todo lo aprendido en clasificación para crear una app real:
un sistema de recomendación de cocinas que corre **en el navegador**.

## El flujo

```
Datos → sklearn → modelo → ONNX → navegador → predicción
```

## El código (Notebook)

```python
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Cargar datos
data = pd.read_csv('../data/cleaned_cuisines.csv')
X = data.iloc[:, 2:]
y = data[['cuisine']]

# Entrenar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = SVC(kernel='linear', C=10, probability=True, random_state=0)
model.fit(X_train, y_train.values.ravel())

# Convertir a ONNX
initial_type = [('float_input', FloatTensorType([None, 380]))]
onx = convert_sklearn(model, initial_types=initial_type)

# Guardar
with open('./model.onnx', 'wb') as f:
    f.write(onx.SerializeToString())
```

## El código (App Web)

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script>
    const ingredients = Array(380).fill(0);
    
    async function startInference() {
        const session = await ort.InferenceSession.create('./model.onnx');
        const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
        const feeds = {};
        feeds[session.inputNames[0]] = input;
        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        alert('Cocina: ' + results[outputName].data[0])
    }
</script>
```

## ¿Qué es ONNX?

| Concepto | Significado |
|----------|-------------|
| **ONNX** | Formato estándar para modelos ML en producción |
| **Tensor** | Estructura de datos que el modelo recibe |
| **Inferencia** | Usar el modelo para predecir |
| **skl2onnx** | Convierte sklearn a ONNX |
| **Onnx Runtime** | Ejecuta modelos ONNX en cualquier plataforma |

## ¿Por qué ONNX en vez de Flask?

| Flask (3-Web-App) | ONNX (4-Applied) |
|-------------------|------------------|
| Necesita servidor Python | Corre en el navegador |
| Requiere internet | Funciona offline |
| Más flexible | Más rápido |

## Errores comunes

| Error | Solución |
|-------|----------|
| 404 model.onnx | Verificar que esté en la misma carpeta |
| ort is not defined | Actualizar CDN de onnxruntime-web |
| Shape mismatch | Verificar número de features (380) |
