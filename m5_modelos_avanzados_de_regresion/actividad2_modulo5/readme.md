# 📈 Aplicación Comparativa de Técnicas Avanzadas de Regresión

Este proyecto implementa y compara tres técnicas avanzadas de regresión en distintos contextos, utilizando datos reales y métricas apropiadas para evaluar el rendimiento de cada enfoque.

## 🚀 Características

- Tres escenarios simulados con datasets reales:
  - Predicción de precios de viviendas con Elastic Net
  - Estimación de percentiles de ingreso con Regresión Cuantílica
  - Proyección de variables macroeconómicas con modelo VAR
- Preprocesamiento adaptado a cada caso
- Entrenamiento, evaluación y visualización de resultados
- Comparación crítica de los enfoques utilizados

## 📁 Estructura del Proyecto

```
actividad2_modulo5/
├── principal.py             # Código completo del proyecto
├── readme.md                # Este archivo
├── Figure_1.png             # Predicciones de Regresión Cuantílica (primeros 100 registros)
├── Figure_2.png             # Proyección VAR - 5 pasos adelante
├── requirements.txt         # Paquetes utilizados
├── capturas_terminal.txt    # Registro de ejecución
```

## ⚙️ Uso del Proyecto

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar el script:
```bash
python principal.py
```

Este script carga los datos, entrena los modelos, evalúa los resultados y muestra visualizaciones clave. También imprime el tiempo total de ejecución.

## 📊 Métricas utilizadas

- RMSE para Elastic Net
- Pérdida Pinball (quantile loss) para regresión cuantílica
- Visualización y comparación de predicciones para VAR

## 📚 Datasets

- **California Housing**: predicción de precios de vivienda (sklearn.datasets)
- **Adult Income**: estimación de percentiles de ingreso (fetch_openml)
- **Macroeconomic Series**: realgdp, realcons y realinv (statsmodels)

## 🧠 Análisis final y comparación crítica

**¿Qué técnica resultó más robusta para cada caso?**  
- **Elastic Net** fue adecuada para problemas con multicolinealidad en datos continuos, como los precios de vivienda.  
- **Regresión Cuantílica** permitió obtener una visión más completa de la distribución de ingresos, modelando distintos percentiles.  
- **VAR** demostró ser útil para modelar relaciones temporales entre múltiples variables económicas.

**¿Qué ventajas o limitaciones presentó cada enfoque?**  
- Elastic Net balancea entre L1 y L2, pero requiere escalado y tuning fino.  
- Regresión Cuantílica es sensible al ruido y más costosa computacionalmente.  
- VAR depende fuertemente de la estacionariedad y del lag óptimo, pero modela relaciones dinámicas entre series.

**¿Qué técnica recomendarías para cada contexto?**  
- **Elastic Net** para regresión tabular con muchas variables correlacionadas.  
- **Regresión Cuantílica** para problemas donde interesa estimar rangos o intervalos.  
- **VAR** para análisis económico multivariable con dependencia temporal.

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia Técnica

Apoyo en depuración de código y documentación por:  
ChatGPT (gpt-4o, build 2025-07).
