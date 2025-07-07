# 🧪 Clasificación Médica con Ajuste de Hiperparámetros

Este proyecto construye un modelo de clasificación para predecir la probabilidad de que un paciente tenga diabetes tipo II, utilizando el dataset *Pima Indians Diabetes*. Se entrena un modelo base con Random Forest y luego se aplican dos técnicas de optimización de hiperparámetros: **Grid Search** y **Random Search**. Finalmente, se comparan métricas de rendimiento y tiempos de ejecución.

---

## 🚀 Características

- Random Forest como modelo base
- Preprocesamiento con imputación, escalado y división 70/30
- Ajuste de hiperparámetros con:
  - Grid Search (exploración exhaustiva)
  - Random Search (exploración aleatoria)
- Evaluación con métricas estándar
- Visualización comparativa y reporte de tiempos de entrenamiento

---

## 📂 Estructura del Proyecto

```
ACTIVIDAD2_MODULO4/
│
├── modelo/
│   ├── cargar_datos.py
│   ├── preprocesamiento.py
│   ├── modelo_base.py
│   ├── ajuste_hiperparametros.py
│   ├── evaluacion.py
│   └── visualizacion.py
│
├── principal.py                  # Ejecuta todo el flujo del proyecto
├── requirements.txt              # Dependencias
├── grafico_comparativo1.jpeg     # Comparación Accuracy y F1 Score
├── grafico_comparativo2.jpeg     # Importancia de características
├── tiempos_entrenamiento.txt     # Tiempos de Grid y Random Search
└── readme.md                     # Este archivo
```

---

## 📥 Uso del Proyecto

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el flujo completo

```bash
python principal.py
```

Este comando:
- Carga y explora los datos
- Preprocesa: reemplazo de ceros, escalado, split 70/30
- Entrena modelo base
- Aplica Grid Search y Random Search
- Evalúa todos los modelos
- Genera comparaciones gráficas
- Guarda métricas y tiempos en archivos

---

## 📊 Métricas utilizadas

- **Accuracy**
- **F1 Score**
- **Recall**
- **AUC (Área bajo la curva ROC)**

---

## 📈 Visualizaciones

1. `grafico_comparativo1.jpeg`: comparación visual entre Accuracy y F1 de los tres modelos.
2. `grafico_comparativo2.jpeg`: importancia de variables según el mejor modelo ajustado.

---

## 📚 Dataset

Dataset utilizado:  
[Pima Indians Diabetes Dataset (CSV)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

Este dataset contiene variables médicas de pacientes mujeres y un indicador binario de si desarrollaron diabetes tipo II.

---

## 🤔 Reflexión final y análisis comparativo

Durante esta actividad se compararon dos métodos clásicos de ajuste de hiperparámetros para el modelo Random Forest: Grid Search y Random Search. Ambos lograron mejorar el rendimiento respecto al modelo base, aunque con diferencias notables en tiempo y precisión.

### ❓ ¿Cuál técnica fue más eficiente?
**Random Search** fue más eficiente en términos de tiempo. Tardó considerablemente menos que Grid Search, lo que es esperable ya que evalúa menos combinaciones.

### 🏆 ¿Cuál encontró el mejor modelo?
**Grid Search** entregó el modelo con mejor desempeño general. Alcanzó una mayor precisión y F1 Score, producto de una exploración más exhaustiva del espacio de hiperparámetros.

### 💡 ¿Qué hubieras hecho diferente?
En una siguiente iteración habría incluido **Optuna** para aplicar optimización bayesiana, que suele ofrecer una mejor relación entre rendimiento y tiempo en datasets medianos o grandes. Además, habría añadido validación cruzada estratificada manualmente para asegurar balance en cada fold.

---

## 👤 Autor

Este proyecto fue desarrollado por **Rubén Schnettler.** 
Viña del Mar, Chile.

---

## 🤖 Asistencia Técnica

Asistencia en depuración de errores y documentación proporcionada por:  
**`ChatGPT (gpt-4o, build 2025-07)`**  

---