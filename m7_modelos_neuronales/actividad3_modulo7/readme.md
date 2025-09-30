# 📘 actividad sesión 3 --- autoencoders (reconstrucción y denoising) con mnist

este proyecto implementa **autoencoders densos** para la tarea de
reconstrucción de imágenes y reducción de ruido usando el dataset
**mnist** (dígitos manuscritos). se comparan dos enfoques:  
- un **autoencoder básico**, que aprende a reconstruir imágenes limpias.  
- un **autoencoder denoising**, que recibe entradas ruidosas y aprende a
recuperar la versión original limpia.  

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion3/`.  
- no requiere descarga manual de datos (usa `tf.keras.datasets.mnist`).  

---

## 📦 estructura del proyecto

```
actividad3_modulo7/
├── principal.py
├── readme.md
└── resultados_sesion3/
    ├── curvas_loss_basico.png
    ├── curvas_loss_denoising.png
    ├── reconstrucciones_basico.png
    ├── denoising_tripleta.png
    ├── informe_resultados.txt
    ├── metricas.json
    ├── modelo_basico_resumen.txt
    ├── modelo_denoising_resumen.txt
    ├── modelo_basico.keras
    └── modelo_denoising.keras
```

---

## 1) dataset y preprocesamiento

- **dataset**: mnist (60.000 imágenes de entrenamiento y 10.000 de
  prueba).  
- **preprocesamiento**: normalización de píxeles al rango [0,1] y
  re-formateo a vectores de tamaño 784.  
- para denoising, se añadió **ruido gaussiano** con desviación estándar
  0.5 a las imágenes de entrada.  

---

## 2) resultados obtenidos

### autoencoder básico

- pérdida en test (bce) = **0.1589**  
- pérdida en test (mse) = **0.0308**  
- el modelo logra reconstrucciones fieles de las imágenes originales.  

(ver `curvas_loss_basico.png` y `reconstrucciones_basico.png`)

### autoencoder denoising ✅ (ligeramente mejor)

- pérdida en test (bce) = **0.1526**  
- pérdida en test (mse) = **0.0291**  
- el modelo reduce efectivamente el ruido y recupera imágenes claras.  

(ver `curvas_loss_denoising.png` y `denoising_tripleta.png`)

---

## 3) análisis

- **arquitectura**: ambos modelos comparten la misma estructura densa,
  con una capa de codificación (64 neuronas) y una decodificación
  simétrica, sumando ~218k parámetros.  
- **curvas de entrenamiento**: se observa convergencia rápida en pocas
  épocas. el uso de *early stopping* evitó sobreajuste y el `clipnorm`
  estabilizó los gradientes en apple silicon, lo que permitió mantener
  pérdidas razonables y entrenamientos reproducibles.  
- **rendimiento cuantitativo**: el autoencoder denoising obtuvo valores
  de pérdida levemente inferiores (bce=0.1526 vs. 0.1589), lo cual
  confirma que entrenar con entradas ruidosas no perjudica el desempeño,
  sino que incluso mejora la generalización.  
- **evaluación cualitativa**: las reconstrucciones del modelo básico son
  nítidas, pero el denoising demuestra un valor añadido: cuando se
  introducen imágenes contaminadas con ruido, las salidas resultan más
  claras y estables, preservando la morfología de los dígitos.  
- **interpretación pedagógica**: este ejercicio muestra que los
  autoencoders no solo comprimen información, sino que también pueden
  aprender funciones útiles como la “limpieza” de señales, lo cual los
  hace aplicables en dominios reales como eliminación de ruido en audio
  o imágenes médicas.  

---

## 4) conclusión

- el **autoencoder básico** es un buen punto de partida y confirma la
  capacidad del modelo para representar de forma comprimida y luego
  reconstruir los dígitos. sin embargo, su objetivo está limitado a
  copiar imágenes limpias.  
- el **autoencoder denoising** resultó ganador, ya que mantiene un
  rendimiento muy cercano en reconstrucción de imágenes limpias, pero
  además aporta la capacidad de restaurar datos degradados. sus métricas
  son ligeramente mejores y las visualizaciones confirman un resultado
  más robusto ante perturbaciones.  
- desde el punto de vista de aprendizaje profundo, esto demuestra la
  importancia de **entrenar con ejemplos difíciles** (en este caso,
  imágenes ruidosas) para lograr modelos más resilientes.  
- a futuro, se podría explorar el uso de **autoencoders convolucionales
  (convnets)**, que aprovechan la estructura espacial de las imágenes y
  suelen producir resultados mucho más nítidos en tareas de denoising.
  también es posible experimentar con técnicas de regularización
  (dropout, weight decay) o con capas de mayor profundidad para mejorar
  aún más la generalización.  

---

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
