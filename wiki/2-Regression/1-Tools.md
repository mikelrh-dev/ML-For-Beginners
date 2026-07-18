# 1. Configurando tu entorno 🛠️

> "Antes de construir una casa, necesitas tus herramientas."

## ¿Qué aprendemos aquí?

Antes de tocar código ML, necesitas configurar tu computadora. 
Es como preparar la mesa antes de cocinar.

## El código

```python
# Verificar versión de Python
import sys
print(sys.version)

# Verificar que las librerías estén instaladas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
```

## Desglose paso a paso

| Librería | ¿Qué hace? | Analogía |
|----------|-------------|----------|
| `pandas` | Manejar datos como tablas | Excel en Python |
| `numpy` | Matemáticas rápidas | Calculadora supercargada |
| `matplotlib` | Hacer gráficas | Lienzo para dibujar datos |
| `seaborn` | Gráficas bonitas | Instagram para gráficas |
| `sklearn` | Modelos de ML | Caja de herramientas ML |

## ¿Por qué esto?

1. **Python 3.8+**: Necesitas una versión reciente
2. **Jupyter**: Tu "cuaderno de laboratorio" interactivo
3. **scikit-learn**: La librería estándar para ML en Python

## Conceptos clave

- **Jupyter Notebook**: Documentos interactivos que mezclan código y texto
- **Librerías**: Paquetes de código reutilizable
- **IDE**: Entorno de desarrollo (VS Code, JupyterLab)

## Errores comunes

```bash
# ❌ Error: "No module named 'pandas'"
# ✅ Solución:
pip install pandas numpy matplotlib seaborn scikit-learn

# ❌ Error: "Python no encontrado"
# ✅ Solución: Instalar Python 3.8+ desde python.org
```

## Siguiente paso

Ahora que tienes tus herramientas, vamos a cargar datos reales → [2-Data](./2-Data.md)
