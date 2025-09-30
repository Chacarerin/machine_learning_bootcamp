# 📘 actividad sesión 5 --- cnn con regularización y optimización (cifar-10)

este proyecto implementa una **red neuronal convolucional (cnn) regularizada** para clasificar las
10 categorías del dataset **cifar-10**. se incorporan técnicas de regularización (**l2** y **dropout**),
además de callbacks de optimización (**early stopping**, **reduceLROnPlateau**), con el objetivo
de mejorar la generalización y controlar el sobreajuste. también se analiza la evolución del
**learning rate** durante el entrenamiento.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion5/`.  
- no requiere descarga manual de datos (usa `tf.keras.datasets.cifar10`).  
- permite ajustar hiperparámetros por línea de comandos (`--optimizador`, `--lr`, `--l2`, `--dropout`, etc.).  

---

## 📦 estructura del proyecto

```
actividad5_modulo7/
├── principal.py
├── readme.md
└── resultados_sesion5/
    ├── curvas_entrenamiento.png
    ├── learning_rate.png
    ├── matriz_confusion.png
    ├── reporte_clasificacion.txt
    ├── resumen.json
    ├── modelo_cnn.keras
    ├── mejor_modelo.keras
    └── modelo_resumen.txt
```

---

## 1) dataset y preprocesamiento

- **dataset**: cifar-10 (50.000 imágenes de entrenamiento y 10.000 de prueba en 10 clases).  
- **preprocesamiento**: normalización de píxeles a rango [0,1].  
- **split de validación**: 10% del set de entrenamiento reservado para validación durante el aprendizaje.  

---

## 2) resultados obtenidos

- **test accuracy** = **0.8077**  
- **test loss** = **0.5964**  

(ver `resumen.json`)

### reporte de clasificación (extracto)

- clases con mejor desempeño:  
  - **auto**: precisión = 0.93, recall = 0.90, f1 = 0.91  
  - **barco**: precisión = 0.86, recall = 0.94, f1 = 0.90  
- clases más débiles:  
  - **gato**: precisión = 0.76, recall = 0.56, f1 = 0.64  
  - **pájaro**: precisión = 0.78, recall = 0.66, f1 = 0.72  

(ver `reporte_clasificacion.txt`)

### visualizaciones

- `curvas_entrenamiento.png`: muestran una reducción progresiva de pérdida y estabilización en validación.  
- `learning_rate.png`: evidencia la disminución del lr gracias a *ReduceLROnPlateau*.  
- `matriz_confusion.png`: refleja un desempeño equilibrado, con mayor confusión entre clases visualmente similares (p. ej. gato ↔ perro, ciervo ↔ caballo).  

---

## 3) análisis

- la cnn propuesta alcanzó un **80.7% de accuracy en test**, superando ampliamente a un clasificador aleatorio (10%) y mostrando un buen equilibrio entre clases.  
- las técnicas de regularización (**l2 y dropout**) ayudaron a evitar sobreajuste, como se aprecia en la cercanía entre curvas de entrenamiento y validación.  
- la dinámica del **learning rate** confirma que la estrategia adaptativa permitió seguir aprendiendo a lo largo de las épocas, aunque la precisión se estabilizó cerca del 81%.  
- la matriz de confusión resalta que las confusiones ocurren principalmente en clases con similitudes visuales, lo que indica que el modelo captura características generales pero aún le cuesta diferenciar detalles finos.  
- el rendimiento es competitivo para una cnn compacta, aunque todavía inferior a arquitecturas más profundas preentrenadas.  

---

## 4) conclusión

- el objetivo de implementar una **cnn regularizada con callbacks de optimización** se cumplió de forma satisfactoria.  
- el modelo logra un **desempeño robusto (~81% de accuracy)**, con buena estabilidad en validación y control del sobreajuste.  
- las métricas muestran fortalezas en clases bien definidas (autos, barcos) y debilidades en aquellas con alta variabilidad intra-clase (gatos, pájaros).  
- para mejorar se recomienda:  
  - aumentar la profundidad de la red o probar arquitecturas preentrenadas (*transfer learning*).  
  - ampliar el número de épocas con un *scheduler* más gradual.  
  - explorar técnicas adicionales de aumento de datos para reforzar clases más complejas.  

en síntesis, la actividad demuestra el impacto positivo de la **regularización y el ajuste dinámico de hiperparámetros** en redes convolucionales aplicadas a cifar-10.

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
