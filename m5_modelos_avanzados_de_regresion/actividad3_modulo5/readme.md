# 🤖 Comparación de Métodos de Boosting y Bagging en Predicción de Ingresos

Este proyecto aplica técnicas de ensamblado (ensemble methods) como Bagging y Boosting para predecir si una persona gana más de $50.000 USD anuales, utilizando el dataset Adult Income.

## 🚀 Características

- Dataset: Adult Income (OpenML)
- Predicción binaria (ingreso >50K)
- Modelos implementados:
  - Random Forest (Bagging)
  - AdaBoost (Boosting)
  - XGBoost (Boosting)
- Evaluación con métricas clásicas:
  - Accuracy
  - Matriz de Confusión
- Comparación visual del rendimiento

## 📁 Estructura del Proyecto

```
actividad3_modulo5/
├── principal.py             # Código completo del proyecto
├── readme.md                # Este archivo
├── requirements.txt         # Paquetes utilizados
├── captura_terminal.txt     # Registro de ejecución
├── Figure_RandomForest.png  # Matriz de confusión RandomForest
├── Figure_AdaBoost.png      # Matriz de confusión AdaBoost
├── Figure_XGBoost.png       # Matriz de confusión XGBoost
├── Figure_Accuracy.png      # Comparación de accuracy
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

El script entrena los modelos, evalúa su rendimiento, genera gráficos y muestra el tiempo total de ejecución.

## 📊 Métricas utilizadas

- Accuracy (Exactitud)
- Matriz de Confusión
- Comparación visual de rendimiento (barra de accuracy)

## 📚 Dataset

El dataset utilizado es el **Adult Income Dataset**, que contiene registros sobre características demográficas y ocupacionales. El objetivo es predecir si una persona gana más de $50.000 USD anuales. Dataset cargado desde OpenML.

Fuente:  
https://www.openml.org/d/1590

## 🧠 Análisis final y comparación crítica

**¿Qué técnica tuvo mejor desempeño?**  
XGBoost obtuvo el mejor rendimiento en términos de accuracy, seguido de Random Forest y luego AdaBoost. En las matrices de confusión se observa que XGBoost tiene un buen balance entre verdaderos positivos y negativos.

**¿Qué ventajas o limitaciones presentó cada enfoque?**  
- **Random Forest** es robusto y fácil de ajustar, pero puede sobreajustar si no se regula bien.
- **AdaBoost** es sensible al ruido y errores mal clasificados.
- **XGBoost** es el más preciso, eficiente y escalable, aunque más complejo de configurar.

**¿Qué modelo recomendarías para producción?**  
Recomiendo **XGBoost**, ya que mostró mayor exactitud y buena generalización. Además, tiene opciones para interpretación como importancia de variables y tolera desequilibrios de clases.

## 🧮 Interpretación de la matriz de confusión

Cada matriz tiene la siguiente estructura:

```
              Predicho
            0        1
Real  0   [TN]     [FP]
      1   [FN]     [TP]
```

- **TN (Verdaderos Negativos)**: correctamente predichos como <=50K  
- **TP (Verdaderos Positivos)**: correctamente predichos como >50K  
- **FP**: clasificados como >50K pero en realidad no lo eran  
- **FN**: clasificados como <=50K pero en realidad sí ganaban más  

El ideal es tener la mayor cantidad posible en la diagonal (TN y TP).

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia Técnica

Apoyo en depuración de código y documentación por:  
ChatGPT (gpt-4o, build 2025-07).
