# 📘 actividad sesión 3 --- clasificación de reseñas clínicas con transformers (hugging face)

este proyecto usa un modelo **transformer multilingüe** para clasificar reseñas clínicas en español.
se emplea el checkpoint **nlptown/bert-base-multilingual-uncased-sentiment** que predice **1–5 estrellas**; luego se agrega la salida a
**polaridad** (*negativo/neutral/positivo*), se compara con un **criterio heurístico** simple y se
generan gráficos y un resumen cuantitativo.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion3/`.  
- acepta `--txt` (una reseña por línea) o `--csv --columna` para entrada personalizada.  
- usa **PyTorch** como backend (se desactiva TensorFlow vía `TRANSFORMERS_NO_TF=1`).  

---

## 📦 estructura del proyecto

```
actividad3_modulo8/
├── principal.py
├── readme.md
└── resultados_sesion3/
    ├── predicciones.csv
    ├── distribucion_estrellas.png
    ├── distribucion_polaridad.png
    └── resumen.json
```

---

## 1) datos y modelo

- **n reseñas** procesadas: **12**  
- **modelo**: `nlptown/bert-base-multilingual-uncased-sentiment` (pipeline de *sentiment-analysis*)  
- **salida primaria**: 1–5 ⭐; **salida agregada**: negativo/neutral/positivo.  

---

## 2) resultados obtenidos

- **recuento por estrellas**: 1: 1, 2: 4, 3: 1, 4: 2, 5: 4  
- **recuento por polaridad**: negativo: 5, neutral: 1, positivo: 6  
- **acuerdo con heurística** (aprox. criterio humano): **91.67%**  
- ver `distribucion_estrellas.png` y `distribucion_polaridad.png` para las distribuciones.  
- `predicciones.csv` contiene: texto, etiqueta cruda del modelo, score, estrellas, polaridad,
  heurístico y si hubo **acuerdo**.  

---

## 3) análisis

- la distribución de estrellas muestra un **balance razonable** entre valoraciones positivas (4–5⭐) y
  negativas (1–2⭐); el agregado por polaridad indica **tendencia positiva** del set.  
- el **acuerdo con el heurístico** es una **medida proxy**: útil para detectar desalineaciones
  obvias (p. ej., textos con términos claramente negativos clasificados como positivos) pero **no sustituye**
  una evaluación humana etiquetada.  
- los **scores** del pipeline permiten priorizar ejemplos de alta confianza para auditoría rápida.  
- como modelo **multilingüe**, maneja bien español general; matices clínicos específicos podrían requerir
  ajuste fino (*fine-tuning*) con datos del dominio.  

---

## 4) conclusión

- se cumple el objetivo de **clasificar reseñas** con un **Transformer** y entregar una
  **síntesis interpretable** (estrellas, polaridad, acuerdo con heurística).  
- el flujo es **reproducible** (un solo script) y deja artefactos listos para informes.  
- próximas mejoras:
  - **fine-tuning** con reseñas clínicas reales etiquetadas.  
  - calibración de umbrales y **detección de conflicto** (score alto + desacuerdo con heurístico).  
  - añadir explicabilidad ligera (palabras salientes) mediante *saliency* o LIME/SHAP.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
