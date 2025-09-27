# 📘 actividad sesión 2 --- clustering con dbscan y hdbscan (make_moons)

este proyecto aplica **clustering basado en densidad** usando **dbscan**
y **hdbscan** sobre un dataset sintético de formas no convexas. el
objetivo es comparar el desempeño de ambos métodos usando métricas
objetivas (**silhouette** y **davies--bouldin**) y visualización en 2d.

------------------------------------------------------------------------

## ▶️ ejecución rápida

``` bash
python principal.py
```

-   genera todas las salidas en `resultados_sesion2/`.\

-   no requiere datasets externos (usa `sklearn.datasets.make_moons`).\

-   para ejecutar hdbscan se requiere instalar:

    ``` bash
    pip install hdbscan
    ```

------------------------------------------------------------------------

## 📦 estructura del proyecto

    actividad_sesion2/
    ├── principal.py
    ├── readme.md
    └── resultados_sesion2/
        ├── dataset_make_moons.png
        ├── dbscan_mejor.png
        ├── hdbscan.png
        ├── dbscan_grid_resultados.csv
        ├── resumen.json
        └── resumen.txt

------------------------------------------------------------------------

## 1) dataset y preprocesamiento

-   **dataset**: `make_moons(n=1000, noise=0.10)`\
-   **preprocesamiento**: escalado con `standardscaler` + reducción con
    `pca` a 2 componentes (para visualización).

------------------------------------------------------------------------

## 2) resultados obtenidos

### dbscan (mejor configuración)

-   **eps=0.3**, **min_samples=10**\
-   silhouette = **0.384**\
-   davies--bouldin = **1.025**\
-   2 clusters principales detectados + ruido en los bordes.

(ver `dbscan_mejor.png`)

### hdbscan

-   **min_cluster_size=15**\
-   silhouette = **0.295**\
-   davies--bouldin = **1.957**\
-   también identificó 2 clusters, pero con menor separación clara y más
    puntos de ruido.

(ver `hdbscan.png`)

------------------------------------------------------------------------

## 3) comparación y análisis

-   **dbscan** obtuvo un índice silhouette mayor (**0.384 vs 0.295**),
    indicando **mejor separación relativa de los clusters**.\
-   **davies--bouldin** también favorece a dbscan (**1.025 vs 1.957**),
    mostrando menor dispersión interna respecto a la separación.\
-   **hdbscan** se comportó de manera estable sin necesidad de ajustar
    eps, pero no alcanzó la misma calidad de agrupamiento que la
    configuración óptima de dbscan en este dataset.

------------------------------------------------------------------------

## 4) conclusión

-   **ganador**: **dbscan** con `eps=0.3`, `min_samples=10`.\
-   dbscan fue más eficaz en este dataset sencillo al capturar la
    estructura de las "lunas" y separar ruido, superando a hdbscan en
    ambas métricas.\
-   sin embargo, **hdbscan** mantiene ventajas en escenarios reales con
    densidades más heterogéneas, donde elegir eps para dbscan es
    complejo.

------------------------------------------------------------------------

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**\
📍 Viña del Mar, Chile.

------------------------------------------------------------------------

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
