# Glosario de Regresión 📖

## Términos clave

| Término | Definición | Analogía |
|---------|------------|----------|
| **Feature** | Variable de entrada (X) | Ingredientes de una receta |
| **Target** | Variable a predecir (y) | El plato que quieres cocinar |
| **Modelo** | Fórmula que aprende de datos | La receta que aprendiste |
| **Training** | Entrenar el modelo | Practicar una receta |
| **Testing** | Evaluar el modelo | Probar si sabe bien |
| **Overfitting** | Modelo memoriza, no aprende | Receta que solo sabe bien en tu cocina |
| **Underfitting** | Modelo muy simple | Receta muy básica que no convence |
| **R² score** | Métrica de regresión (0-1) | Calificación del plato |
| **Accuracy** | Métrica de clasificación (%) | % de platos exitosos |
| **Label** | Categoría objetivo | Nombre del plato |
| **Epoch** | Una pasada completa por los datos | Una vuelta completa de mezclar |
| **Loss** | Error del modelo | Cuánto se aparta del sabor ideal |

## Tipos de regresión

| Tipo | ¿Cuándo? | Ejemplo |
|------|----------|---------|
| **Lineal** | Predecir número continuo | Precio de casa |
| **Logística** | Predecir categoría | Spam o no spam |

## Funciones de sklearn

| Función | Propósito |
|---------|-----------|
| `train_test_split()` | Dividir datos en train/test |
| `LinearRegression()` | Modelo de regresión lineal |
| `LogisticRegression()` | Modelo de clasificación |
| `LabelEncoder()` | Convertir categorías a números |
| `model.fit()` | Entrenar el modelo |
| `model.predict()` | Hacer predicciones |
| `model.score()` | Evaluar precisión |

## Errores comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `ConvergenceWarning` | Modelo no convergió | `max_iter=1000` |
| `ValueError: could not convert` | Strings sin codificar | `LabelEncoder` |
| `FileNotFoundError` | Path incorrecto | Verificar ruta |
| `KeyError` | Nombre de columna mal | `df.columns` |
