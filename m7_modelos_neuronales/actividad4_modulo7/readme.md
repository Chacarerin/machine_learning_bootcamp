# 📘 actividad sesión 4 --- transfer learning con efficientnetb0 (cifar-10)

este proyecto implementa un enfoque de **transfer learning** usando
**efficientnetb0** (preentrenado en imagenet) aplicado al dataset
**cifar-10**. el objetivo fue comprobar la utilidad de un modelo
preentrenado en un dataset generalista al clasificar imágenes pequeñas y
de dominio distinto. se evaluó el desempeño mediante curvas de
entrenamiento, matriz de confusión, reporte de clasificación y métricas
globales.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion4/`.  
- no requiere descarga manual de datos (usa `tf.keras.datasets.cifar10`).  
- se puede cambiar de backbone con `--modelo resnet` o activar *fine-tuning* con `--fine_tune`.  

---

## 📦 estructura del proyecto

```
actividad4_modulo7/
├── principal.py
├── readme.md
└── resultados_sesion4/
    ├── curvas_entrenamiento_fase1.png
    ├── matriz_confusion.png
    ├── predicciones_vs_reales.png
    ├── reporte_clasificacion.txt
    ├── resumen.json
    ├── modelo_efficientnet.keras
    └── modelo_efficientnet_resumen.txt
```

---

## 1) dataset y preprocesamiento

- **dataset**: cifar-10 (50.000 imágenes de entrenamiento y 10.000 de
  prueba en 10 clases).  
- **preprocesamiento**: reescalado de 32x32 a 224x224 píxeles,
  normalización con la función propia de efficientnet y aumento de datos
  (flip horizontal, rotación, zoom).  

---

## 2) resultados obtenidos

### métricas globales

- **test accuracy** = **0.1001**  
- **test loss** = **2.3011**  

(ver `resumen.json`)

### reporte de clasificación (extracto)

como puede observarse en `reporte_clasificacion.txt`, el modelo
prácticamente no logró distinguir las clases. el único resultado notable
fue en la clase **“perro”** con recall = 1.0 (todas las imágenes fueron
predichas como perro), pero esto redujo el rendimiento general en el
resto de categorías.

### visualizaciones

- `curvas_entrenamiento_fase1.png`: muestran pérdida estable pero sin
  mejora de exactitud.  
- `matriz_confusion.png`: evidencia la predicción dominante en una sola
  clase.  
- `predicciones_vs_reales.png`: la mayoría de ejemplos terminan
  malclasificados.  

---

## 3) análisis

- **transferencia fallida**: el modelo entrenado con efficientnetb0 no
  logró generalizar en cifar-10 bajo la configuración usada. el
  rendimiento se quedó en el nivel de un clasificador aleatorio (10% de
  accuracy).  
- **causas posibles**:  
  - falta de entrenamiento suficiente (solo 10 épocas, sin *fine-tuning*
    del backbone).  
  - tamaño reducido de imágenes de cifar-10 (32x32) que, al reescalar a
    224x224, genera interpolaciones que pueden degradar la información.  
  - necesidad de ajustar hiperparámetros como learning rate, batch size,
    o aplicar *fine-tuning* parcial en capas superiores.  
- **comparación con expectativas**: se esperaba al menos un 60–70% de
  accuracy con transfer learning. los resultados muestran que congelar
  completamente el backbone en este caso fue insuficiente para aprender
  patrones útiles.  
- **valor pedagógico**: aunque el resultado numérico fue bajo, el
  ejercicio demuestra de manera clara las limitaciones de usar modelos
  preentrenados sin adaptación al dominio del dataset objetivo. esto
  permite comprender la importancia del *fine-tuning* y de ajustar la
  estrategia de transferencia.  

---

## 4) conclusión

- el uso directo de **efficientnetb0** preentrenado no resultó efectivo
  en cifar-10 con la configuración inicial. el modelo colapsó en una sola
  clase (perro), logrando apenas un 10% de accuracy.  
- para mejorar, se recomienda:  
  - habilitar **fine-tuning** de al menos las últimas capas del
    backbone.  
  - probar con **resnet50** para comparar desempeños.  
  - entrenar con más épocas y usar *schedulers* de tasa de aprendizaje.  
  - experimentar con arquitecturas convolucionales diseñadas para
    imágenes pequeñas (como convnets custom o resnet reducidas).  
- este resultado responde al objetivo de la actividad: se realizó la
  implementación, se obtuvieron métricas y visualizaciones, y se analizó
  críticamente por qué el transfer learning no alcanzó un buen
  desempeño.  

---

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
