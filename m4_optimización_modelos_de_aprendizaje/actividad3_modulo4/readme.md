# 🧠 Optimización Bayesiana en Clasificación Médica

Este proyecto implementa un modelo de clasificación binaria usando Random Forest para predecir la presencia de cáncer de mama. Se utilizan dos enfoques de optimización bayesiana para ajustar hiperparámetros: Scikit-Optimize y Hyperopt, evaluando su impacto en el rendimiento y eficiencia del modelo.

## 🚀 Características

- Uso del dataset de cáncer de mama (Scikit-learn)
- Escalado de variables con StandardScaler
- División de datos en entrenamiento y prueba (70/30)
- Entrenamiento de modelo base sin optimización
- Optimización de hiperparámetros con:
  - Scikit-Optimize (BayesSearchCV)
  - Hyperopt (TPE)
- Comparación de métricas y tiempos de ejecución

## 📂 Estructura del Proyecto

```
ACTIVIDAD3_MODULO4/
├── principal.py               # Contiene todo el código del proyecto
├── requirements.txt           # Paquetes utilizados
├── captura_terminal.txt       # Evidencia de ejecución completa
└── readme.md                  # Este archivo
```

## 📥 Uso del Proyecto

1. Instalar dependencias:
pip install -r requirements.txt

2. Ejecutar el proyecto:
python principal.py

Este comando ejecuta todo el flujo del proyecto:
- Carga y escala los datos
- Entrena modelo base
- Aplica Scikit-Optimize y luego Hyperopt
- Muestra métricas y tiempos de cada método

## 📊 Métricas utilizadas

- F1 Score
- Tiempo de ejecución
- Classification Report

## 📚 Dataset

Se utiliza el dataset load_breast_cancer de Scikit-learn, el cual contiene características de imágenes de tumores de mama y un indicador binario que clasifica como maligno o benigno.

## 🤔 Reflexión final y análisis comparativo

¿Cuál técnica fue más eficiente?  
Hyperopt fue significativamente más rápida (menos de 3 segundos) en comparación con Scikit-Optimize (más de 12 segundos), alcanzando el mismo F1-Score. Esto la convierte en una opción más eficiente para este tipo de problemas.

¿Cuál entregó el mejor resultado?  
Ambas técnicas entregaron exactamente el mismo F1-Score (0.9772), pero Scikit-Optimize tardó más. En este caso, no hubo diferencia en calidad de modelo, solo en eficiencia.

¿Qué aprendiste del proceso?  
Aprendí que distintas bibliotecas pueden implementar el mismo enfoque de optimización con diferentes resultados en tiempo. Además, confirmé que la optimización bayesiana es útil para reducir la cantidad de combinaciones necesarias y aún así obtener un muy buen desempeño.

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia Técnica

Optimización de código y documentación proporcionada por:  
ChatGPT (gpt-4o, build 2025-07).