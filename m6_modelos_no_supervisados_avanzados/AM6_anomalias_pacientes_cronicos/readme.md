# 📘 Evaluación Modular — Módulo 6: Segmentación y Detección de Anomalías en Pacientes Crónicos

Este proyecto aplica técnicas de aprendizaje no supervisado para segmentar pacientes crónicos en grupos con características similares e identificar casos atípicos que podrían requerir atención especial.

---

## 📌 Características del proyecto

- Dataset: Datos clínicos de pacientes crónicos (normalizados y preprocesados).
- Tarea: Segmentación y detección de anomalías.
- Modelos aplicados:
  - **DBSCAN** para segmentación con ruido.
  - **HDBSCAN** como variante jerárquica para patrones complejos.
  - **Isolation Forest** y **One-Class SVM** para detección de anomalías.
- Evaluación del modelo:
  - Índice de Silueta.
  - Índice Davies-Bouldin.
  - Porcentaje de ruido detectado.
- Visualizaciones:
  - Reducción de dimensionalidad (`pca.png`, `tsne.png`, `umap.png`).
  - Clústeres detectados (`clusters_dbscan_*.png`, `clusters_hdbscan_*.png`).
  - Anomalías (`anomalias_*_*.png`).
- Reportes:
  - `metricas_clustering.csv`
  - `anomalias_isolation_forest.csv`
  - `anomalias_oneclass_svm.csv`
  - `analisis_cruzado.csv`
- Tiempo de ejecución reportado.

---

## 🧪 Métricas obtenidas

Modelo: **DBSCAN**  
- Número de clústeres: **4**  
- Porcentaje de ruido: **87.37 %**  
- Índice de Silueta: **0.1306**  
- Índice Davies-Bouldin: **1.2989**

Modelo: **HDBSCAN**  
- Número de clústeres: **0**  
- Porcentaje de ruido: **100 %**  
- (No se reportan métricas válidas)

⏱ Tiempo total de ejecución: **5.61 segundos**

---

## 📁 Estructura del proyecto

```
evaluacion_modulo6/
├── principal.py                     # Código principal completo y comentado
├── readme.md                        # Este documento
├── requirements.txt                 # Librerías utilizadas
├── pca.png, tsne.png, umap.png      # Reducción de dimensionalidad
├── clusters_dbscan_*.png            # Visualización de clústeres DBSCAN
├── clusters_hdbscan_*.png           # Visualización de clústeres HDBSCAN
├── anomalias_*_*.png                # Visualización de anomalías
├── metricas_clustering.csv          # Métricas de evaluación
├── anomalias_isolation_forest.csv   # Anomalías detectadas por Isolation Forest
├── anomalias_oneclass_svm.csv       # Anomalías detectadas por One-Class SVM
└── analisis_cruzado.csv             # Comparación entre modelos
```

---

## 📊 Interpretación de resultados

- **DBSCAN** detectó 4 clústeres, con **alto porcentaje de ruido (87.37 %)**, lo que sugiere que gran parte de los datos no se ajusta a agrupaciones densas claras.  
- El **índice de silueta (0.1306)** es bajo, indicando poca cohesión y separación entre clústeres.  
- El **índice Davies-Bouldin (1.2989)** refuerza que los clústeres no son compactos ni bien diferenciados.  
- **HDBSCAN** no logró detectar clústeres válidos y consideró el **100 % como ruido**, evidenciando la dificultad de segmentar con los parámetros actuales y/o con la estructura de los datos.

---

## 🧠 Análisis de anomalías

- Los detectores **Isolation Forest** y **One-Class SVM** generaron listados de observaciones atípicas (`anomalias_isolation_forest.csv`, `anomalias_oneclass_svm.csv`).  
- Estos resultados permiten priorizar revisión de pacientes con perfiles clínicos inusuales.  
- El archivo `analisis_cruzado.csv` permite identificar **outliers persistentes** (pacientes detectados como anómalos en más de un método), aumentando la confianza en los hallazgos.

---

## 💡 Reflexión final

- Se cumplieron los objetivos del módulo: aplicar **clustering** y **detección de anomalías** en un dataset de pacientes crónicos.  
- La segmentación con DBSCAN y HDBSCAN mostró limitaciones, dado el alto ruido y los bajos valores de silueta.  
- Aun así, la **detección de anomalías** entrega resultados clínicamente relevantes para identificar pacientes de riesgo.  
- Próximos pasos recomendados:
  - Ajustar parámetros de DBSCAN/HDBSCAN (`eps`, `min_samples`, `min_cluster_size`).  
  - Evaluar otras técnicas (Gaussian Mixture Models, Spectral Clustering).  
  - Incorporar más variables clínicas para mejorar la separabilidad.  
  - Validar los hallazgos con especialistas médicos.  

**En resumen:** aunque la segmentación no fue concluyente, la detección de anomalías se constituye en el principal aporte de este análisis, útil para focalizar recursos en pacientes críticos.

---

## 👤 Autor

Este proyecto fue desarrollado por Rubén Schnettler.  
Viña del Mar, Chile.

## 🤖 Asistencia técnica

Apoyo en estructuración y documentación por:  
ChatGPT (gpt-5, 2025).
