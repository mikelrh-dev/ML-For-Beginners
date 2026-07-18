# 2. Cargando y limpiando datos 📊

> "Los datos sucios producen conclusiones sucias."

## ¿Qué aprendemos aquí?

Aquí es donde los datos crudos se convierten en algo útil.
Como lavar y pelar verduras antes de cocinar.

## El código

```python
import pandas as pd

# Cargar datos
df = pd.read_csv('../data/pumpkins.csv')

# Ver las primeras filas
df.head()

# Información general
df.info()

# Limpiar datos
df = df.dropna()
df = df[df['Size'] != 'unknown']
```

## Desglose paso a paso

| Paso | Código | ¿Por qué? |
|------|--------|------------|
| Cargar | `pd.read_csv()` | Leer el archivo CSV |
| Explorar | `df.head()` | Ver qué hay adentro |
| Info | `df.info()` | Tipos de datos y nulos |
| Limpiar | `df.dropna()` | Quitar filas vacías |
| Filtrar | `df[df['Size'] != 'unknown']` | Quitar valores raros |

## ¿Por qué esto?

1. **Calidad de datos**: "Garbage in, garbage out"
2. **Explorar primero**: Siempre mira tus datos antes de modelar
3. **Limpiar**: Los datos reales siempre tienen problemas

## Conceptos clave

- **DataFrame**: Tabla de datos (como Excel)
- **NaN**: Datos faltantes
- **Head**: Primeras filas
- **Dropna**: Eliminar valores nulos

## Errores comunes

```python
# ❌ FileNotFoundError
pd.read_csv('pumpkins.csv')  # Path incorrecto
# ✅ Solución: Verificar la ruta

# ❌ KeyError: 'columna'
df['Columna']  # Nombre mal escrito
# ✅ Solución: df.columns para ver nombres reales
```

## Siguiente paso

Ahora que tienes datos limpios, vamos a graficarlos → [3-Linear](./3-Linear.md)
