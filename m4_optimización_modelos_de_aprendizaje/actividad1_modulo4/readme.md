# 🧪 Predicción de Diabetes tipo II con Random Forest y Ajuste de Hiperparámetros

Este proyecto implementa un modelo de clasificación en Python para predecir la presencia de **diabetes tipo II** utilizando el conocido dataset *Pima Indians Diabetes*. Se utiliza un modelo **Random Forest** como base, y se aplican técnicas de **ajuste de hiperparámetros** para mejorar su rendimiento. Además, se generan métricas de evaluación y gráficos para facilitar la interpretación del modelo.

---

## 🚀 Características del Proyecto

- Clasificación binaria con Random Forest
- Preprocesamiento del dataset: limpieza, escalado, partición entrenamiento/prueba
- Ajuste de hiperparámetros con:
  - Grid Search
  - Random Search
- Visualización de importancia de características
- Comparación de rendimiento entre modelos
- Evaluación con F1 Score, Precisión, Recall y AUC

---

## 📂 Estructura del Proyecto

```
ACTIVIDAD1_MODULO4/
│
├── modelo/                       # Subcarpeta con módulos funcionales
│   ├── cargar_datos.py
│   ├── preprocesamiento.py
│   ├── modelo_base.py
│   ├── ajuste_hiperparametros.py
│   ├── evaluacion.py
│   └── visualizacion.py
│
├── principal.py                  # Script principal que ejecuta todo el flujo
├── requirements.txt              # Lista de dependencias del proyecto
├── grafico_comparativo1.jpeg     # Comparación de F1 Score
├── grafico_comparativo2.jpeg     # Importancia de características
└── readme.md                     # Este archivo
```

---

## 📥 Cómo usar este proyecto

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/actividad1_modulo4.git
cd actividad1_modulo4
```

### 2. Instala las dependencias

```bash
pip install -r requirements.txt
```

> Recomendación: usa un entorno virtual para evitar conflictos con otras instalaciones de Python.

### 3. Ejecuta el proyecto

```bash
python principal.py
```

Este script:

- Carga y explora el dataset desde la web
- Realiza el preprocesamiento (limpieza, imputación, escalado)
- Entrena un modelo base
- Aplica Grid Search y Random Search
- Evalúa y compara modelos con métricas estándar
- Genera gráficos en archivos `.jpeg`

---

## 📈 Visualizaciones

- `grafico_comparativo1.jpeg`: compara el F1 Score del modelo base, Grid Search y Random Search.
- `grafico_comparativo2.jpeg`: muestra la importancia de cada característica según el mejor modelo ajustado.

---

## 📊 Métricas utilizadas

- **F1 Score**
- **Precisión**
- **Recall**
- **AUC (Área bajo la curva ROC)**

Estas métricas permiten evaluar la efectividad del modelo considerando tanto falsos positivos como negativos, lo cual es fundamental en el diagnóstico médico.

---

## 📚 Dataset

El proyecto utiliza el dataset *Pima Indians Diabetes*, disponible públicamente:

📄 [Descargar dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

---

## 🤔 Reflexión

El ajuste de hiperparámetros es una herramienta poderosa para mejorar el rendimiento de los modelos de machine learning. En este proyecto, el uso de Grid Search y Random Search demostró cómo puede optimizarse un modelo base como Random Forest para obtener mejores resultados en métricas como el F1 Score y el AUC.

Aunque Grid Search obtuvo los mejores resultados, fue también el más costoso computacionalmente. En cambio, Random Search demostró ser una alternativa más rápida y bastante competitiva. En proyectos de mayor escala, se recomienda el uso de métodos más eficientes como **Optimización Bayesiana (Optuna)**.

Este trabajo también demuestra la importancia de desarrollar proyectos de machine learning de forma modular, reutilizable y con documentación clara para facilitar su comprensión y ejecución por terceros.

---

## 👤 Autor

Este proyecto fue desarrollado por **Rubén Schnettler.**

---

## 🤖 Asistencia Técnica

Durante el desarrollo se recibió apoyo en:

- Diagnóstico de errores de ejecución
- Redacción técnica y elaboración del presente `README.md`

**Asistencia proporcionada por:** `ChatGPT (gpt-4o, build 2025-07)`  
**Modo de ayuda:** Depuración de código y documentación.

---