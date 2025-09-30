# 📘 actividad sesión 1 --- nlp tradicional en notas clínicas

este proyecto aplica técnicas de **procesamiento de lenguaje natural (nlp) clásico** sobre un
conjunto de **10 notas clínicas simuladas**. el objetivo es explorar representaciones basadas en
**bag of words** y **tf-idf**, calcular **similaridad coseno** entre documentos y analizar los
términos más relevantes de cada texto.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion1/`.  
- no requiere datasets externos (se usa un corpus de ejemplo definido en el código, editable).  
- admite como entrada un `.txt` propio con un documento por línea (`--corpus_txt`).  

---

## 📦 estructura del proyecto

```
actividad1_modulo8/
├── principal.py
├── readme.md
└── resultados_sesion1/
    ├── heatmap_similaridad.png
    ├── top_terminos_doc_0.png
    ├── top_terminos_doc_1.png
    ├── top_terminos_doc_2.png
    ├── bow_matrix_shape.txt
    ├── tfidf_matrix_shape.txt
    ├── similaridad_coseno_matrix.txt
    ├── corpus_limpio.json
    └── resumen.json
```

---

## 1) dataset y preprocesamiento

- **dataset**: 10 notas clínicas simuladas.  
- **limpieza**: conversión a minúsculas, eliminación de puntuación y espacios múltiples.  
- **vectorización**: bag of words y tf-idf con n-gramas hasta bigramas.  
- **matrices resultantes**:  
  - bow: 10 x 208  
  - tf-idf: 10 x 208  

(ver `bow_matrix_shape.txt` y `tfidf_matrix_shape.txt`)  

---

## 2) resultados obtenidos

- la matriz de similaridad coseno (10x10) refleja bajos niveles de similitud global, como es esperable
  en un corpus clínico variado.  
- algunos pares con mayor similitud:  
  - doc3 ↔ doc7 (0.095)  
  - doc1 ↔ doc4 (0.084)  
  - doc5 ↔ doc7 (0.085)  
- el análisis de términos con mayor peso tf-idf permitió identificar los síntomas o diagnósticos más
  distintivos de cada nota (ver `top_terminos_doc_*.png`).  

(ver `heatmap_similaridad.png`, `similaridad_coseno_matrix.txt`, `resumen.json`)  

---

## 3) análisis

- el **heatmap de similaridad** confirma que las notas clínicas son en su mayoría independientes,
  con similitudes puntuales cuando comparten patrones de síntomas (p. ej. dolor o fiebre).  
- el uso de **tf-idf** destaca términos clínicos clave como *cefalea*, *glicemia*, *ecg* o *helicobacter*,
  lo que facilita discriminar rápidamente el tema central de cada documento.  
- la metodología es útil para **detectar grupos de casos relacionados** y **extraer palabras relevantes**
  sin necesidad de modelos complejos.  
- las métricas cuantitativas (similaridades ~0.06–0.09) muestran que el corpus es heterogéneo, lo cual es
  adecuado para evaluar la capacidad de las representaciones de resaltar similitudes reales.  

---

## 4) conclusión

- el objetivo de **aplicar técnicas clásicas de nlp (bow, tf-idf, coseno)** fue alcanzado.  
- se logró representar un corpus clínico breve, medir similitudes y resaltar términos característicos.  
- los resultados demuestran que, incluso con métodos básicos, es posible obtener insights útiles en un
  dominio especializado como la salud.  
- futuras mejoras podrían incluir:  
  - ampliar el corpus para detectar clusters más claros.  
  - incorporar lematización y eliminación de stopwords específicas del dominio médico.  
  - comparar con embeddings modernos (*word2vec*, *bert*) para observar ganancias en representación.  

en síntesis, la actividad evidencia el valor de los enfoques clásicos como primer paso en análisis de
texto clínico.  

---

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
