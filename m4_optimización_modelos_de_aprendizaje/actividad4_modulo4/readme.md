# 🧬 Optimización de Hiperparámetros con Algoritmos Genéticos

Este proyecto aplica un algoritmo genético usando la librería DEAP para ajustar los hiperparámetros de un modelo `RandomForestClassifier`, utilizando el dataset de cáncer de mama (`load_breast_cancer`) de Scikit-learn. Se evalúa la calidad del modelo optimizado frente a la versión base sin ajuste.

## 🚀 Características

- Dataset: Breast Cancer Wisconsin de Scikit-learn
- Escalado de variables con StandardScaler
- División 70/30 en entrenamiento y prueba
- Entrenamiento de modelo base sin optimización
- Optimización de hiperparámetros usando:
  - Algoritmo genético implementado con DEAP
  - 15 generaciones mínimas
  - Población inicial de 10 individuos
  - Métrica de evaluación: F1 Score (cross-validation)
- Comparación de resultados base vs optimizados

## 📂 Estructura del Proyecto

ACTIVIDAD4_MODULO4/
├── principal.py               # Código completo del proyecto
├── requirements.txt           # Paquetes utilizados
├── captura_terminal.txt       # Evidencia de ejecución
└── readme.md                  # Este archivo

## 📥 Uso del Proyecto

1. Instalar dependencias:
pip install -r requirements.txt

2. Ejecutar el proyecto:
python principal.py

Este comando:
- Carga y escala los datos
- Entrena modelo base sin optimización
- Ejecuta el algoritmo genético con DEAP
- Evalúa y reporta el modelo optimizado
- Compara ambos modelos

## 📊 Métricas utilizadas

- F1 Score
- Classification Report
- Comparación final en test set

## 📚 Dataset

Se utiliza el dataset `load_breast_cancer` de Scikit-learn, que contiene variables clínicas asociadas a tumores de mama, clasificando entre malignos y benignos.

## 🤔 Reflexión final y análisis comparativo

¿Cuál técnica fue más eficiente?  
El algoritmo genético logró encontrar una combinación de hiperparámetros que mejoró el modelo base, utilizando solo una fracción de las combinaciones posibles. Es especialmente útil cuando el espacio de búsqueda es grande o no conviene una búsqueda exhaustiva.

¿Se mejoró el rendimiento respecto al modelo base?  
Sí. El F1-Score del modelo optimizado fue superior al del modelo base, lo que indica que la selección evolutiva fue efectiva para este tipo de problema.

¿Son los algoritmos genéticos una buena alternativa?  
Sí, especialmente cuando se necesita balance entre exploración y rendimiento. Aunque no garantizan encontrar el óptimo global, en la práctica logran resultados competitivos con menos recursos computacionales comparados con Grid Search.

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia Técnica

Depuración de código y documentación proporcionada por:  
ChatGPT (gpt-4o, build 2025-07).