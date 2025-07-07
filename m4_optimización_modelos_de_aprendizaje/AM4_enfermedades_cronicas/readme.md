
# 🧠 Proyecto ML: Diagnóstico de Enfermedades Crónicas

Este proyecto implementa un modelo de clasificación supervisada para diagnosticar enfermedades crónicas a partir de síntomas codificados. Utiliza técnicas de preprocesamiento, modelado base con Random Forest y optimización de hiperparámetros con **Optuna** y **Ray Tune**. 

---

## 📂 Estructura del Proyecto

```
AM4_ENFERMEDADES_CRONICAS/
│
├── data/                      # Archivos CSV: entrenamiento, prueba y particiones
│   ├── Training.csv
│   ├── Testing.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── depuraciones_py/                 # Scripts Python del proyecto
│   ├── preparacion.py
│   ├── modelo_base.py
│   ├── optimizacion_optuna.py
│   └── optimizacion_raytune.py
│
├── capturas_terminal/         # Evidencias gráficas
│   ├── terminal_preparacion_py.png
│   ├── terminal_modelo_base_py.png
│   ├── terminal_optimizacion_optuna_py.txt
│   └── terminal_optimizacion_raytune_py.txt
```

---

## 🧪 Dataset

- 4920 observaciones totales.
- 132 variables binarias de entrada (síntomas).
- 1 variable categórica (`prognosis`) con **41 clases balanceadas** (120 casos por clase).
- Fuente: Kaggle (dataset sintético para fines educativos).

---

## ⚙️ Procesamiento de Datos

Archivo: `preparacion.py`

- Se aplicó codificación de etiquetas (`LabelEncoder`) sobre la variable objetivo.
- División en conjunto de entrenamiento (80%) y prueba (20%).
- Resultados:
  - `X_train`: (3936, 132)
  - `X_test`: (984, 132)
  - `y_train`: (3936,)
  - `y_test`: (984,)

---

## 🌲 Modelo Base: Random Forest

Archivo: `modelo_base.py`

- Modelo: `RandomForestClassifier()` con parámetros por defecto.
- Resultados:
  - Accuracy: **1.0**
  - F1-score (macro y weighted): **1.0**
  - ROC AUC Score: **1.0**
  - Todas las clases fueron clasificadas correctamente.
- Matriz de confusión completamente diagonal.

---

## 🔍 Optimización con Optuna

Archivo: `optimizacion_optuna.py`

- Se definió una función objetivo sobre 4 hiperparámetros clave:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Se ejecutaron 30 *trials* con validación cruzada.
- Mejores hiperparámetros:
```python
{'n_estimators': 74, 'max_depth': 23, 'min_samples_split': 5, 'min_samples_leaf': 1}
```
- Accuracy en test: **1.0**

---

## ⚡ Optimización con Ray Tune

Archivo: `optimizacion_raytune.py`

- Se utilizó `ASHAScheduler` para búsqueda eficiente.
- Se ejecutaron 20 *trials*.
- Mejores hiperparámetros:
```python
{'n_estimators': 173, 'max_depth': 28, 'min_samples_split': 2, 'min_samples_leaf': 4}
```
- Accuracy en test: **1.0**

> Nota: Ray mostró advertencias relacionadas con versiones (`tune.report()` vs `session.report()`), pero no afectaron los resultados.

---

## 🤖 Uso de Asistencia con IA

Durante el desarrollo se utilizó **ChatGPT-4** como herramienta de soporte para:

- Depurar errores en tiempo de ejecución.
- Adaptar código entre librerías como Optuna y Ray Tune.
- Diagnosticar errores comunes como rutas relativas, conflictos entre procesos o argumentos duplicados.
- Redactar este `README.md` y estructurar el informe técnico.

---

## 📌 Conclusiones

- El dataset presentaba una estructura perfectamente balanceada y altamente diferenciada entre clases, lo cual facilitó la clasificación con una precisión perfecta (Accuracy = 1.0).
- Ambos métodos de optimización (Optuna y Ray Tune) confirmaron la robustez del modelo Random Forest en este escenario.
- Se recomienda contrastar estos resultados con un conjunto de datos real y ruidoso para evaluar generalización y evitar sobreajuste.

---

## 🚀 Cómo ejecutar

```bash
# Recomendado: usar entorno virtual
pip install -r requirements.txt

# Preprocesamiento
python modelo_py/preparacion.py

# Entrenar modelo base
python modelo_py/modelo_base.py

# Optimización con Optuna
python modelo_py/optimizacion_optuna.py

# Optimización con Ray Tune
python modelo_py/optimizacion_raytune.py
```

---

## 📎 Requisitos

- Python 3.12+
- pandas, scikit-learn, optuna, ray[tune]

---

## 📸 Capturas de terminal

Las salidas de terminal se encuentran dentro de la carpeta `capturas_terminal/` como respaldo para entrega académica o revisión.

---

## 👤 Autor

Rubén Schnettler – Bootcamp Machine Learning  
Región de Valparaíso, Chile
