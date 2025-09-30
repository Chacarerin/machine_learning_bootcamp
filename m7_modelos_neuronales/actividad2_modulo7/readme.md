# 📘 actividad sesión 2 --- rnn (lstm) y red generativa adversaria (gan)

este proyecto implementa dos enfoques complementarios de aprendizaje profundo:  
1. una **red recurrente (lstm)** para la **clasificación de sentimientos** en el dataset **imdb**.  
2. una **red generativa adversaria (gan)** para la **generación de imágenes sintéticas** del dataset **mnist**.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion2/`.  
- no requiere descarga manual de datos (usa `tf.keras.datasets.imdb` y `tf.keras.datasets.mnist`).  

---

## 📦 estructura del proyecto

```
actividad2_modulo7/
├── principal.py
├── readme.md
└── resultados_sesion2/
    ├── imdb_curvas.png
    ├── imdb_matriz_confusion.png
    ├── imdb_resumen.txt
    ├── gan_curvas.png
    ├── gan_resumen.txt
    ├── resumen.txt
    └── gan_iter_*.png
```

---

## 1) dataset y preprocesamiento

### imdb
- **dataset**: imdb (50.000 reseñas de películas, etiquetadas como positivas o negativas).  
- **preprocesamiento**: tokenización y padding de secuencias de texto.  

### mnist
- **dataset**: mnist (60.000 imágenes de entrenamiento, 10.000 de prueba).  
- **preprocesamiento**: normalización de píxeles a rango [0,1].  

---

## 2) resultados obtenidos

### parte a – imdb con lstm ✅ (mejor desempeño en clasificación)

- **test accuracy = 0.7580**  
- **test loss = 0.5547**  
- **matriz de confusión**: tn=9742, fp=2758, fn=3291, tp=9209  

(ver `imdb_curvas.png` y `imdb_matriz_confusion.png`)

### parte b – gan para mnist

- **iteraciones = 3000**  
- **batch size = 128**  
- el generador logra producir dígitos sintéticos reconocibles tras el entrenamiento.  

(ver `gan_curvas.png` y `gan_iter_*.png`)

---

## 3) análisis

- la **lstm aplicada a imdb** logra un desempeño aceptable en clasificación de sentimientos (≈76% de accuracy). el análisis de la matriz de confusión muestra un equilibrio razonable entre verdaderos positivos y negativos, aunque con espacio de mejora en los falsos negativos.  
- la **gan en mnist** evidencia la capacidad de confrontación entre generador y discriminador, mejorando progresivamente la calidad visual de los dígitos. con 3000 iteraciones, ya se obtienen resultados claros y estables.  

---

## 4) conclusión

- **clasificación**: la lstm se consolida como una arquitectura efectiva para tareas de procesamiento de lenguaje natural, aunque con margen de mejora en optimización de hiperparámetros.  
- **generación**: la gan demuestra el poder de los modelos generativos para crear ejemplos sintéticos, siendo un buen punto de partida para futuras aplicaciones en dominios más complejos.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
