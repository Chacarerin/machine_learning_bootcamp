# 📘 Comparación de Modelos con Regularización (Lasso, Ridge y ElasticNet)

Este proyecto aplica técnicas de regularización sobre un modelo de regresión para mejorar la generalización y comparar el impacto de Lasso, Ridge y ElasticNet en los coeficientes del modelo.

## 📌 Características del proyecto

- Dataset: Adult Income (OpenML)
- Tarea: Clasificación binaria (ingreso >50K)
- Modelos aplicados:
  - Lasso
  - Ridge
  - ElasticNet
- Evaluación con:
  - RMSE (Root Mean Squared Error)
  - Análisis comparativo de coeficientes
- Visualización:
  - Gráfico de comparación de coeficientes entre modelos

## 📁 Estructura del proyecto

```
actividad5_modulo5/
├── principal.py                # Código completo y comentado
├── readme.md                   # Documentación del proyecto
├── requirements.txt            # Librerías utilizadas
├── Figure_Coeficientes.png     # Gráfico comparativo de coeficientes
├── capturas_terminal.txt       # Registro de ejecución
```

## 🧪 Métricas utilizadas

- **RMSE**: mide el error de predicción promedio
- **Coeficientes**: muestran la importancia o peso de cada variable en la predicción

Se usó un conjunto de prueba del 20% para evaluar el rendimiento de cada modelo.

## 🔎 Resultados

Ejemplo de salida obtenida en consola:

```
Entrenando modelo: Lasso
Lasso - RMSE: 0.3684

Entrenando modelo: Ridge
Ridge - RMSE: 0.3652

Entrenando modelo: ElasticNet
ElasticNet - RMSE: 0.3669

Resumen de errores RMSE:
Lasso: 0.3684
Ridge: 0.3652
ElasticNet: 0.3669

Tiempo total de ejecución: 6.43 segundos
```

Se genera además una figura con la comparación de los coeficientes de los tres modelos.

## 📊 Interpretación de los coeficientes

- **Lasso** tiende a reducir muchos coeficientes a cero, eliminando variables menos relevantes.
- **Ridge** conserva todas las variables, pero reduce los coeficientes.
- **ElasticNet** combina ambos enfoques: elimina algunas variables y reduce otras.

El gráfico `Figure_Coeficientes.png` permite observar estas diferencias de forma clara.

## 💡 Conclusión

- **Ridge** obtuvo el menor RMSE, aunque por una diferencia pequeña.
- **Lasso** eliminó varias variables, lo cual puede ser útil para simplificar el modelo.
- **ElasticNet** mostró un equilibrio entre reducción y eliminación de coeficientes.

**Modelo recomendado:** `ElasticNet`, por su capacidad de regularizar sin eliminar tanta información y por ofrecer un buen desempeño general.

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia técnica

Apoyo en estructuración y documentación por:  
ChatGPT (gpt-4o, 2025).
