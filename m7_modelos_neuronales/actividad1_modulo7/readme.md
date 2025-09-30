# 📘 actividad sesión 1 --- clasificador de imágenes con redes neuronales (fashion mnist)

este proyecto implementa una **red neuronal densa** para clasificar las
10 categorías del dataset **fashion mnist**. se exploran distintas
combinaciones de funciones de pérdida (**categorical crossentropy, mse**)
y optimizadores (**adam, sgd**), evaluando su impacto en la precisión y
la pérdida sobre el conjunto de prueba.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion1/`.  
- no requiere descarga manual de datos (usa `tf.keras.datasets.fashion_mnist`).  

---

## 📦 estructura del proyecto

```
actividad1_modulo7/
├── principal.py
├── readme.md
└── resultados_sesion1/
    ├── curvas_categorical_crossentropy_adam.png
    ├── curvas_categorical_crossentropy_sgd.png
    ├── curvas_mse_adam.png
    ├── curvas_mse_sgd.png
    ├── reporte_categorical_crossentropy_adam.txt
    ├── reporte_categorical_crossentropy_sgd.txt
    ├── reporte_mse_adam.txt
    ├── reporte_mse_sgd.txt
    ├── resumen.json
    └── resumen.txt
```

---

## 1) dataset y preprocesamiento

- **dataset**: fashion mnist (60.000 imágenes de entrenamiento,
  10.000 de prueba).  
- **preprocesamiento**: normalización de píxeles a rango [0,1] +
  one-hot encoding de etiquetas.  

---

## 2) resultados obtenidos

### combinación 1: categorical crossentropy + adam ✅ (mejor)

- test accuracy = **0.8635**  
- test loss = **0.3833**  
- muestra curvas estables con buena convergencia.  

(ver `curvas_categorical_crossentropy_adam.png`)

### combinación 2: categorical crossentropy + sgd

- test accuracy = **0.8258**  
- test loss = **0.4980**  

(ver `curvas_categorical_crossentropy_sgd.png`)

### combinación 3: mse + adam

- test accuracy = **0.8606**  
- test loss = **0.0202**  
- rendimiento cercano al mejor, aunque la métrica de pérdida no es tan
  intuitiva para clasificación.  

(ver `curvas_mse_adam.png`)

### combinación 4: mse + sgd

- test accuracy = **0.6563**  
- test loss = **0.0509**  
- desempeño significativamente inferior en comparación con las demás.  

(ver `curvas_mse_sgd.png`)

---

## 3) análisis

- la combinación **categorical crossentropy + adam** alcanzó la mayor
  precisión en test (**86.3%**).  
- el uso de **sgd** fue menos efectivo que adam, especialmente combinado
  con mse, donde el accuracy cayó a ~65%.  
- mse como función de pérdida no es la más adecuada para clasificación
  multiclase, aunque con adam logró resultados aceptables.  
- categorical crossentropy demostró ser más estable y consistente para
  este problema.  

---

## 4) conclusión

- **ganador**: **categorical crossentropy + adam**  
- esta configuración logra el mejor balance entre precisión y
  estabilidad en el entrenamiento.  
- para mejorar, podrían explorarse arquitecturas más profundas,
  regularización adicional o más épocas de entrenamiento.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
