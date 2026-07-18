# 1. Creando una app web con tu modelo 🌐

> "El mejor modelo es el que la gente puede usar."

## ¿Qué aprendemos aquí?

Convertimos nuestro modelo de ML en una aplicación web real.
Como pasar de cocinar en casa a abrir un restaurante.

## El código

### Paso 1: Preparar el modelo

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Cargar y limpiar datos
df = pd.read_csv('data/ufos.csv')
df = df.dropna()
df = df[df['Seconds'] <= 60]
df = df[df['Country'].isin(['US', 'GB', 'CA'])]

# Preparar features
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])

X = df[['Seconds', 'Country', 'Latitude', 'Longitude']]
y = df['Country']

# Entrenar modelo
model = LogisticRegression()
model.fit(X, y)

# Guardar modelo
pickle.dump(model, open('ufo_model.p', 'wb'))
```

### Paso 2: Crear app Flask

```python
# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('ufo_model.p', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    seconds = int(request.form['seconds'])
    country = int(request.form['country'])
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    
    prediction = model.predict([[seconds, country, latitude, longitude]])
    return render_template('index.html', prediction=prediction[0])
```

## Desglose paso a paso

| Paso | Archivo | ¿Qué hace? |
|------|---------|-------------|
| 1. Entrenar | `notebook.ipynb` | Crear modelo ML |
| 2. Guardar | `ufo_model.p` | Modelo serializado |
| 3. Flask | `app.py` | Servidor web |
| 4. HTML | `templates/index.html` | Interfaz de usuario |

## ¿Por qué esto?

1. **pickle**: Serializar objetos Python (guardar el modelo)
2. **Flask**: Framework web mínimo y simple
3. **Formularios HTML**: Recibir datos del usuario
4. **API**: Endpoint `/predict` que recibe datos y devuelve predicción

## Conceptos clave

- **Serialización**: Guardar objeto Python como archivo
- **Template**: HTML con variables dinámicas
- **Endpoint**: URL que procesa requests
- **POST**: Método HTTP para enviar datos

## Errores comunes

```python
# ❌ FileNotFoundError: 'ufo_model.p'
# Modelo no guardado
# ✅ Ejecutar celda de pickle primero

# ❌ ConvergenceWarning
# Modelo no convergió
# ✅ Usar max_iter=1000

# ❌ CORS error
# Flask sin CORS configurado
# ✅ pip install flask-cors
```

## Flujo completo

```
Usuario → HTML Form → Flask → Modelo → Predicción → HTML
```

## Siguiente paso

Ahora tienes un modelo en producción. Vamos a explorar conceptos clave → [Glossary](../2-Regression/glossary.md)
