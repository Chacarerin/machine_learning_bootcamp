# 📘 Validación Cruzada con Modelos de Clasificación

Este proyecto evalúa distintas estrategias de validación cruzada utilizando un modelo de regresión logística aplicado al dataset Adult Income.

## 📌 Características del proyecto

- Dataset: Adult Income (OpenML)
- Tarea: Clasificación binaria (ingreso >50K)
- Modelos aplicados:
  - Logistic Regression (`max_iter=1000`)
- Estrategias de validación cruzada:
  - K-Fold
  - Stratified K-Fold
  - Leave-One-Out (con subset de 500 registros)
- Evaluación con:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Matriz de confusión
  - Curva ROC
  - Curva Precision-Recall

## 📁 Estructura del proyecto

```
actividad4_modulo5/
├── principal.py                  # Código completo y comentado
├── readme.md                     # Documentación del proyecto
├── requirements.txt              # Librerías utilizadas
├── Figure_Matriz_KFold.png
├── Figure_ROC_KFold.png
├── Figure_PR_KFold.png
├── Figure_Matriz_StratifiedKFold.png
├── Figure_ROC_StratifiedKFold.png
├── Figure_PR_StratifiedKFold.png
├── Figure_Matriz_LeaveOneOut.png
├── Figure_ROC_LeaveOneOut.png
├── Figure_PR_LeaveOneOut.png
```

## 🧪 Métricas utilizadas

Se utilizaron las siguientes métricas de evaluación:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC (Área bajo la curva ROC)

Cada técnica de validación fue aplicada sobre el mismo modelo, y sus resultados fueron comparados.

## 🔎 Evaluación de resultados

A continuación, un ejemplo de los resultados obtenidos impresos en consola al ejecutar el código:

```
Evaluando con KFold...
KFold - Accuracy: 0.8390, Precision: 0.7216, Recall: 0.5707, F1-Score: 0.6373

Evaluando con StratifiedKFold...
StratifiedKFold - Accuracy: 0.8369, Precision: 0.7214, Recall: 0.5568, F1-Score: 0.6285

Evaluando con LeaveOneOut...
LeaveOneOut - Accuracy: 0.8340, Precision: 0.7255, Recall: 0.5736, F1-Score: 0.6407
```

Se generan automáticamente las matrices de confusión y curvas ROC y PR para cada técnica, guardadas en archivos PNG.

## 💡 Conclusión

- Todas las técnicas entregan resultados consistentes en términos de desempeño.
- `StratifiedKFold` mantiene el balance de clases en cada pliegue, lo que mejora la estabilidad de los resultados.
- `LeaveOneOut` entrega métricas sólidas, pero es más costosa en términos computacionales.
- `KFold` es eficiente pero sensible al desbalance de clases.

**Recomendación:** Usar `StratifiedKFold` por su estabilidad y precisión sin sobrecargar el sistema.

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia técnica

Apoyo en documentación y depuración de código por:  
ChatGPT (gpt-4o, 2025).
