# 📘 actividad sesión 3 --- reducción de dimensionalidad con pca (iris)

este proyecto aplica **análisis de componentes principales (pca)** sobre
el dataset clásico **iris**. el objetivo es reducir de 4 a 2 dimensiones,
interpretar la varianza explicada y visualizar la proyección en 2d,
evaluando además el impacto de pca en un clasificador simple (knn).

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion3/`.  
- no requiere datasets externos (usa `sklearn.datasets.load_iris`).  

---

## 📦 estructura del proyecto

```
actividad_sesion3/
├── principal.py
├── readme.md
└── resultados_sesion3/
    ├── 01_varianza_explicada.png
    ├── 02_pca_2d.png
    ├── pca_scores_pc12.csv
    ├── pca_componentes_pc12.csv
    ├── resumen.json
    ├── resumen.txt
    └── metricas_knn.txt
```

---

## 1) dataset y preprocesamiento

- **dataset**: iris (150 observaciones, 4 variables).  
- **preprocesamiento**: escalado con `standardscaler` + reducción con
  `pca` a 2 componentes.  

---

## 2) resultados obtenidos

### varianza explicada

- **pc1 = 0.7296**  
- **pc2 = 0.2285**  
- **acumulada pc1+pc2 = 0.9581**  

(ver `01_varianza_explicada.png`)

### proyección 2d

- la visualización en 2 componentes muestra **una clara separación entre
  las especies de iris**, especialmente setosa, mientras que versicolor
  y virginica presentan cierta superposición.  

(ver `02_pca_2d.png`)

### comparación knn con y sin pca

- **accuracy test sin pca = 0.9211**  
- **accuracy test con pca = 0.8947**  
- **cv(5) promedio sin pca = 0.9600**  
- **cv(5) promedio con pca = 0.9133**  

(ver `metricas_knn.txt`)

---

## 3) análisis

- las **dos primeras componentes principales retienen ~96% de la
  varianza**, lo que permite una representación 2d muy informativa.  
- la separación entre clases mejora en términos de **visualización** y
  exploración, aunque al usar pca como preprocesamiento en knn se observa
  una **ligera pérdida de precisión** (0.92 → 0.89 en test).  
- pca resulta útil para **reducir dimensionalidad y ruido** en datasets
  más grandes o con muchas variables, incluso si en iris no mejora
  necesariamente el rendimiento predictivo.  

---

## 4) conclusión

- pca en iris confirma que **dos componentes son suficientes** para
  capturar la mayor parte de la variabilidad (95.8%).  
- la **visualización en 2d** permite interpretar patrones y agrupamientos
  entre especies.  
- como paso de preprocesamiento, **pca puede simplificar el espacio sin
  degradar mucho la performance** de modelos simples como knn.  

---

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

Documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
