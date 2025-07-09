# 🧠 Comparación de Métodos de Optimización de Hiperparámetros con Ray Tune

Este proyecto aplica dos estrategias automáticas para la optimización de hiperparámetros de un modelo `RandomForestClassifier`, utilizando el dataset de cáncer de mama de Scikit-learn. Se comparan las metodologías **Random Search** y **Grid Search** implementadas mediante la librería `Ray Tune`, evaluando sus desempeños con la métrica F1 Score.

---

## 🚀 Características

- Dataset: Breast Cancer Wisconsin de Scikit-learn
- Preprocesamiento con `StandardScaler` y división 70/30
- Entrenamiento de modelo base sin optimización
- Optimización de hiperparámetros con:
  - Random Search (búsqueda aleatoria)
  - Grid Search (búsqueda exhaustiva)
- Métrica de evaluación: F1 Score
- Comparación final de modelos optimizados vs base

---

## 📂 Estructura del Proyecto

```
ACTIVIDAD5_MODULO4/
├── principal.py             # Código completo del proyecto
├── requirements.txt         # Paquetes utilizados
├── captura_terminal.txt     # Evidencia de ejecución
└── readme.md                # Este archivo
```

---

## 📥 Uso del Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/actividad5_modulo4.git
cd actividad5_modulo4
```

> Reemplaza `tu-usuario` con tu nombre de usuario real en GitHub.

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el proyecto

```bash
python principal.py
```

Esto ejecutará:
- La carga y preprocesamiento de datos
- El entrenamiento del modelo base
- La búsqueda automática de hiperparámetros con Ray Tune
- La evaluación de cada configuración
- El reporte final con los mejores resultados

---

## 📦 requirements.txt

Este proyecto requiere únicamente los siguientes paquetes:

```text
numpy==2.3.1
pandas==2.3.0
scikit-learn==1.7.0
ray==2.47.1
```

---

## 📊 Métricas utilizadas

- F1 Score (validación cruzada)
- Classification Report sobre test set
- Comparación entre modelos base, Random Search y Grid Search

---

## 📚 Dataset

Se utiliza el dataset `load_breast_cancer` de Scikit-learn, que contiene variables clínicas asociadas a tumores de mama. El objetivo es predecir si el tumor es benigno o maligno a partir de 30 atributos numéricos.

---

## 🤔 Reflexión final y análisis comparativo

**¿Cuál técnica fue más eficiente?**  
Random Search fue más rápida y ligera computacionalmente, ideal cuando se quiere una solución aproximada sin agotar recursos.

**¿Se mejoró el rendimiento respecto al modelo base?**  
Sí, ambas técnicas lograron mejorar el F1 Score frente al modelo inicial. Grid Search logró una ligera ventaja en rendimiento, aunque con mayor tiempo de cómputo.

**¿Es útil aplicar estrategias de ajuste como estas?**  
Definitivamente. Permiten explorar múltiples combinaciones sin intervención manual, logrando mejoras significativas con relativamente poco esfuerzo adicional.

---

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

---

## 🤖 Asistencia Técnica

Depuración de código y resolución de errores con **GitHub Copilot.** Documentación por 
**ChatGPT (gpt-4o, build 2025-07).**