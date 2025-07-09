# 🧠 Optimización de Modelos de Aprendizaje Automático - Módulo 4

Este repositorio contiene el desarrollo completo de las actividades del **Módulo 4** del curso de *Machine Learning*, centradas en el **ajuste y optimización de modelos de clasificación binaria** utilizando técnicas como búsqueda aleatoria, búsqueda en malla, optimización bayesiana y algoritmos genéticos.

Cada carpeta corresponde a una sesión del módulo, incluyendo también la evaluación final integradora. Los modelos están orientados a la predicción de enfermedades, usando como base el conjunto de datos *Breast Cancer Wisconsin* de Scikit-learn, así como un dataset clínico multivariable disponible en Kaggle.

---

## 📂 Estructura del repositorio

```
.
├── actividad1_modulo4/     # Búsqueda aleatoria y en malla con Ray Tune
├── actividad2_modulo4/     # Visualización y comparación de tuning
├── actividad3_modulo4/     # Optimización Bayesiana con skopt e Hyperopt
├── actividad4_modulo4/     # Algoritmos genéticos con DEAP
├── actividad5_modulo4/     # Comparación de Ray Tune y Optuna
└── evaluacion_modular/     # Proyecto final con dataset clínico (Kaggle)
```

Cada directorio contiene:
- `principal.py`: archivo principal ejecutable
- `requirements.txt`: dependencias mínimas para correr cada proyecto
- `readme.md`: documentación específica por actividad

---

## 🚀 Requisitos generales

Antes de ejecutar cualquier actividad, se recomienda crear un entorno virtual e instalar las dependencias específicas de cada carpeta:

```bash
pip install -r requirements.txt
```

---

## 🔍 Temas abordados

- **Búsqueda aleatoria y en malla (Ray Tune)**
- **Optimización bayesiana (scikit-optimize, Hyperopt)**
- **Algoritmos genéticos (DEAP)**
- **Tuning automático (Optuna vs Ray Tune)**
- **Evaluación de métricas: F1-Score, clasificación binaria**
- **Visualización de resultados y reflexión crítica**

---

## 👤 Autor

Este conjunto de actividades fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.

---

## 🤖 Asistencia técnica

Depuración de código, documentación y resolución de errores con **GitHub Copilot** y  
**ChatGPT (gpt-4o, build 2025-07)**