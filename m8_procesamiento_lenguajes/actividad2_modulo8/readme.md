# 📘 actividad sesión 2 --- preprocesamiento (spaCy/NLTK) + tf-idf en notas clínicas

este proyecto aplica un **pipeline clásico de nlp** sobre un corpus breve de notas clínicas.
se realiza **limpieza**, **tokenización y lematización/stemming**, eliminación de **stopwords**,
y se vectoriza con **tf-idf** (hasta bi-gramas). se visualizan **términos más relevantes por documento**
y se compara **corpus original vs preprocesado** mediante métricas simples (longitud media, vocabulario, ttr).

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion2/`.  
- acepta un `.txt` propio con un documento por línea mediante `--corpus_txt`.  
- usa **spaCy** si el modelo `es_core_news_sm` está disponible; en caso contrario, **fallback** con **NLTK/regex**.  

---

## 📦 estructura del proyecto

```
actividad2_modulo8/
├── principal.py
├── readme.md
└── resultados_sesion2/
    ├── comparacion_original_vs_preprocesado.png
    ├── top_terminos_doc_0.png
    ├── top_terminos_doc_1.png
    ├── top_terminos_doc_2.png
    ├── tfidf_matrix_shape.txt
    ├── vocabulario.json
    ├── corpus_limpio.json
    └── resumen.json
```

---

## 1) dataset y preprocesamiento

- **dataset**: 10 notas clínicas simuladas.  
- **limpieza**: minúsculas, remoción de e‑mails, urls, números aislados y signos de puntuación.  
- **tokenización y lematización/stemming**: intenta **spaCy**; si no está disponible, usa **NLTK**.  
- **stopwords**: de spaCy o NLTK; si fallan, conjunto mínimo de respaldo.  
- **vectorización**: **TF‑IDF** con n‑gramas (1–2).  

según `resumen.json`, el entorno usó fallback ( `spacy_activado = false` ).

---

## 2) resultados obtenidos

### dimensiones de matrices

- `tfidf` = **10 × 77–208** según vocabulario final (ver `tfidf_matrix_shape.txt` y `vocabulario.json`).  

### comparación corpus (original vs preprocesado)

- **longitud media por documento**: **13.8 → 9.9**  
- **tamaño de vocabulario**: **97 → 77**  
- **type‑token ratio (ttr)**: **0.703 → 0.778**  (aumenta distintividad relativa)  

(ver `comparacion_original_vs_preprocesado.png` y `resumen.json`).

### términos más relevantes (ejemplos)

- `doc0`: infección, congestión, nasal, sospecha, viral…  
- `doc1`: abdominal, historial, persistente, gastritis, femenina…  
- `doc2`: saturación, disnea, tórax, radiografía, covid…  

(ver `top_terminos_doc_*.png`).

---

## 3) análisis

- el **preprocesamiento** reduce ruido (números, conectores y signos), lo que se refleja en la caída de
  **longitud media** (13.8→9.9) y **vocabulario** (97→77). pese a tener menos tokens, la **ttr sube**
  (0.703→0.778), indicando mayor **densidad informativa** por palabra.  
- los **términos top por documento** capturan entidades/síntomas clave (p. ej. *gastritis*, *radiografía*, *covid*),
  útiles para **búsqueda semántica simple**, **agrupación** o como rasgos para **clasificadores**.  
- el fallback con **NLTK** mantiene el flujo cuando spaCy no está disponible, conservando resultados coherentes
  para un **baseline** reproducible y liviano.  

---

## 4) conclusión

- se cumple el objetivo de **preprocesar** y **vectorizar** un corpus clínico corto con TF‑IDF, generando
  artefactos interpretables y listos para siguientes tareas (clustering/clasificación).  
- el pipeline mejora la **señal semántica** al eliminar ruido y concentrar términos relevantes; esto se evidencia
  en el aumento del **TTR** y en la claridad de los **términos top**.  
- para una siguiente iteración se sugiere:
  - habilitar **spaCy** con `es_core_news_sm` para una **lematización** más precisa.  
  - extender a **tri‑gramas** o agregar un **diccionario médico** de stopwords.  
  - comparar con **embeddings** (word2vec/fastText/BERT) para evaluar ganancias en tareas downstream.  

---

## 👤 autor

Este proyecto fue desarrollado por **rRubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
