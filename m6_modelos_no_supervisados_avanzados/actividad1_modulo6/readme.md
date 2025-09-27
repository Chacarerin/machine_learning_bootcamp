# 📘 actividad sesión 1 — clustering jerárquico y reducción de dimensionalidad (iris & wine)

este proyecto aplica **clustering jerárquico aglomerativo (ward)** y reducción de dimensionalidad (**pca** y **t-sne**) sobre los datasets clásicos **iris** y **wine**. el objetivo es explorar estructuras latentes sin etiquetas, visualizando dendrogramas y comparando diferentes valores de k.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion1/`.  
- no requiere datasets externos (usa `sklearn.datasets`).  

---

## 📦 estructura del proyecto

```
actividad_sesion1/
├── principal.py
├── readme.md
└── resultados_sesion1/
    ├── dendrograma_iris.png
    ├── clusters_pca_iris.png
    ├── clusters_tsne_iris.png
    ├── resumen_iris.json
    ├── dendrograma_wine.png
    ├── clusters_pca_wine.png
    ├── clusters_tsne_wine.png
    └── resumen_wine.json
```

---

## 1) datasets y resumen

### iris  
- 150 observaciones, 4 variables.  
- agrupamientos:  
  - k=2 → {0: 101, 1: 49}  
  - k=3 → {0: 71, 1: 49, 2: 30}  

### wine  
- 178 observaciones, 13 variables.  
- agrupamientos:  
  - k=2 → {0: 122, 1: 56}  
  - k=3 → {0: 58, 1: 56, 2: 64}  

---

## 2) dendrogramas

- **iris**: muestra separación clara de 3 grupos; un corte en ~3 refleja mejor la estructura.  
- **wine**: la separación es más difusa; se observan tres bloques principales aunque con solapamiento.  

(ver `dendrograma_iris.png` y `dendrograma_wine.png`).  

---

## 3) visualización con reducción de dimensionalidad

- **iris**:  
  - **pca**: separa muy bien *setosa*; versicolor y virginica aparecen más solapadas.  
  - **t-sne**: refuerza la separación en tres grupos compactos.  

- **wine**:  
  - **pca**: muestra tres nubes con intersección parcial.  
  - **t-sne**: logra destacar fronteras locales y clusters más definidos, aunque persiste cierto solapamiento.  

(ver `clusters_pca_*.png` y `clusters_tsne_*.png`).  

---

## 4) interpretación de resultados

- el clustering jerárquico **detecta estructura latente** especialmente clara en iris (k=3).  
- en wine, los grupos son más complejos: la química de los vinos genera **solapamientos** que requieren métodos adicionales (dbscan, umap).  
- **pca** preserva varianza lineal global, útil para interpretación.  
- **t-sne** enfatiza **relaciones locales**, mostrando clusters más compactos.  

---

## 5) reflexión final

- el ejercicio muestra cómo el clustering jerárquico, combinado con reducción de dimensionalidad, permite **exploración visual y analítica** de datasets.  
- las diferencias entre iris y wine reflejan cómo la complejidad del dominio impacta en la calidad de los agrupamientos.  
- se recomienda en escenarios reales complementar con métricas cuantitativas (silhouette, davies-bouldin) y comparar con otros métodos de clustering.  

---

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**  
