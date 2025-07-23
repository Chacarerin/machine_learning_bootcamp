# 📘 Evaluación Modular — Módulo 5: Scoring Crediticio con Interpretabilidad

Este proyecto construye un modelo de clasificación binaria para predecir la probabilidad de buen comportamiento crediticio y, al mismo tiempo, garantizar la interpretabilidad del modelo mediante herramientas de análisis como SHAP.

---

## 📌 Características del proyecto

- Dataset: `credit-g` (OpenML)
- Tarea: Clasificación binaria (`good` vs `bad`)
- Modelo aplicado:
  - Regresión Logística con regularización L1 (Lasso)
- Interpretabilidad:
  - Visualización de importancia de variables con SHAP
- Evaluación del modelo:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC
  - Matriz de confusión
- Visualizaciones:
  - Matriz de confusión (`matriz_confusion.png`)
  - SHAP summary plot (`shap_summary_plot.png`)
- Tiempo de ejecución reportado

---

## 🧪 Métricas obtenidas

```
Accuracy: 0.8050
Precision: 0.8400
Recall: 0.8936
F1 Score: 0.8660
AUC: 0.8188
Tiempo total de ejecución: 0.40 segundos
```

---

## 📁 Estructura del proyecto

```
evaluacion_modulo5/
├── principal.py                # Código principal completo y comentado
├── readme.md                   # Este documento
├── requirements.txt            # Librerías utilizadas
├── capturas_terminal.txt       # Registro de ejecución
├── matriz_confusion.png        # Matriz de confusión
├── shap_summary_plot.png       # Visualización SHAP global
```

---

## 📊 Interpretación de resultados

- El modelo alcanzó un F1 Score de **0.866**, lo que representa un equilibrio adecuado entre precisión y exhaustividad.
- El **recall de 0.8936** indica una muy buena capacidad de identificar correctamente los casos positivos (clientes buenos).
- La **AUC de 0.8188** refleja una sólida capacidad discriminativa del modelo.

---

## 🧠 Análisis de interpretabilidad

- Se utilizó **SHAP (SHapley Additive exPlanations)** para interpretar el modelo.
- El gráfico `shap_summary_plot.png` muestra qué variables tienen mayor impacto positivo o negativo sobre la probabilidad de que un cliente sea "good".
- Dado que se utilizó Lasso, algunas variables con bajo aporte fueron descartadas automáticamente (coeficiente = 0).

---

## 💡 Reflexión final

- El modelo entrega métricas robustas con alta interpretabilidad, ideal para entornos donde las decisiones deben ser justificadas.
- La elección de regresión logística con regularización L1 permite un modelo explicativo, compacto y confiable.
- SHAP aporta valor al traducir los coeficientes a un lenguaje visual y comprensible.

**En resumen:** se logró un modelo predictivo sólido, interpretable y alineado con las exigencias del módulo y la práctica profesional.

---

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia técnica

Apoyo en estructuración y documentación por:  
ChatGPT (gpt-4o, 2025).
