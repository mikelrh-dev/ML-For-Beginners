# Lección 1: Visualización de datos de música nigeriana

## ¿Qué aprendemos aquí?

Antes de aplicar clustering, necesitamos **entender nuestros datos**. Esta lección es como mirar el terreno antes de construir una casa — si no conocés la tierra, vas a tener problemas después.

## El dataset

Tenemos 530 canciones nigerianas de Spotify con 16 columnas:
- **Texto**: nombre, álbum, artista, género
- **Números**: popularidad, bailabilidad, acústica, energía, ruido, tempo, etc.

## ¿Qué hicimos?

### 1. Exploración básica
```
df.info()      → 530 canciones, 16 columnas, sin nulos
df.describe()  → distribución de cada columna
```

### 2. Limpieza
- Eliminamos géneros 'Missing' (Spotify no los clasificó)
- Nos quedamos con 3 géneros: afro dancehall, afropop, nigerian pop
- Eliminamos canciones con popularidad 0 (ruido)

### 3. Visualización
- **Heatmap de correlación**: solo energy-loudness tiene correlación fuerte
- **Gráfico KDE**: muestra dónde se agrupan las canciones
- **Scatter plot**: revela superposición entre géneros

## Resultado clave

Los tres géneros se **superponen bastante** en popularidad y bailabilidad. Esto significa que K-Means va a tener dificultades para separarlos — ¡eso es lo que exploramos en la lección 2!

## Analogía

Es como mirar un mapa de temperaturas antes de decidir dónde poner aire acondicionado. Si todos los colores están mezclados, no vas a poder crear zonas claras de temperatura.

---

**Siguiente:** [K-Means](../2-K-Means/README.md) — Ahora sí aplicamos el algoritmo
