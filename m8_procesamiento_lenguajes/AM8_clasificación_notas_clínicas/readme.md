# 🩺 evaluación modular — módulo 8: clasificación de notas clínicas con enfoque ético y mitigación de sesgos

este proyecto implementa un sistema de nlp para clasificar **notas clínicas** por **gravedad** (`leve`, `moderado`, `severo`) a partir de texto libre y metadatos simples (edad, género, afección). se comparan dos enfoques: **naive bayes + tf‑idf** y **transformer en español (bert-wwm)**, se evalúan **métricas por clase**, se analiza **sesgo** por género y edad, y se incluyen **explicaciones** con lime.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- por defecto lee `dataset_clinico_simulado_200.csv` del mismo directorio.  
- resultados en la carpeta `resultados_mod8/` (gráficos, reportes `.json` y `.csv`).

---

## 📦 estructura

```
actividad_modulo8/
├── principal.py
├── dataset_clinico_simulado_200.csv
└── readme.md
```

---

## 1) datos y eda

- columnas: `texto_clinico`, `edad`, `genero`, `afeccion`, `gravedad`.  
- distribución de clases balanceada:  
  - **leve**: 59  
  - **moderado**: 82  
  - **severo**: 59  
- ver gráfico `clases_distribucion.png`.  
- se generan **grupos etarios** (`<=29`, `30-44`, `45-64`, `65+`) para análisis de sesgos.

---

## 2) preprocesamiento

- **tokenización + limpieza** (minúsculas, remoción de palabras muy cortas y stopwords en español).  
- **stemming** con `snowball` (si está disponible).  
- **tf‑idf** con n‑gramas (1,2) y `min_df=2`.  
- **embeddings word2vec** entrenados localmente sobre el propio corpus (se guardan ejemplos en `ejemplo_embeddings_w2v.npy`).

---

## 3) modelado

- **modelo 1**: `naive_bayes_tfidf` (pipeline tf‑idf + multinomial nb).  
- **modelo 2**: `transformer_es` (bert wwm en español), si `transformers` está instalado.  

en esta ejecución se reporta principalmente **naive bayes**.

---

## 4) resultados obtenidos

**métricas globales (naive bayes):**

| clase     | precisión | recall | f1   | soporte |
|-----------|-----------|--------|------|---------|
| leve      | 1.00      | 1.00   | 1.00 | 15      |
| moderado  | 1.00      | 1.00   | 1.00 | 20      |
| severo    | 1.00      | 1.00   | 1.00 | 15      |
| **macro** | 1.00      | 1.00   | 1.00 | 50      |
| **accuracy global** | **1.00** | | | |

> el modelo logró **100% de exactitud en test**. esto puede deberse a que el dataset es simulado y relativamente sencillo; se recomienda validación con más datos.

**gráficos generados:**  
- `matriz_confusion_nb.png`  
- `clases_distribucion.png`

---

## 5) explicabilidad con lime

ejemplos de palabras más influyentes:

- caso 13 (severo): términos como *“infarto”*, *“agudo”*, *“miocardio”* favorecen la predicción de severidad.  
- caso 30 (leve): *“malestar”*, *“días”*, *“tenido”* se asocian a la clase leve.  
- caso 39 (moderado): *“síntomas”*, *“migraña”*, *“clínico”* impulsan la clasificación moderada.

archivos: `lime_ejemplo_13.txt`, `lime_ejemplo_30.txt`, `lime_ejemplo_39.txt`.

---

## 6) sesgos y equidad

- **género**: el reporte (`sesgo_genero.csv`) muestra valores macro-f1 muy similares entre hombres y mujeres → no se detectan sesgos relevantes.  
- **edad**: (`sesgo_grupo_edad.csv`) desempeño consistente en todos los grupos etarios.  

> aunque no se detectaron diferencias, se recomienda monitoreo constante en datos reales.

---

## 7) reflexión ética

- **riesgos**: sobreajuste en dataset pequeño, reproducción de sesgos lingüísticos, uso indebido de un modelo automatizado como sustituto de criterio médico.  
- **medidas**: auditorías periódicas, validación externa, revisión de variables sensibles, explicación local de predicciones dudosas.  
- **responsabilidad**: el sistema debe ser **apoyo clínico**, nunca reemplazo de la decisión profesional.

---

## 8) conclusiones

- `naive_bayes_tfidf` alcanzó métricas perfectas en este dataset simulado, pero esto exige **cautela y validación externa**.  
- los resultados de lime son coherentes: el modelo usa términos médicos relevantes.  
- los análisis de sesgo no muestran inequidades, pero se requiere más volumen y diversidad de datos para confirmarlo.  
- se valida la importancia de combinar **modelos predictivos + explicabilidad + análisis de sesgos** para un uso ético y responsable.


> Nota: aunque no se entrenó un modelo Transformer por limitaciones de entorno, se utilizaron dos enfoques distintos de representación y modelado (TF-IDF + Naive Bayes y embeddings Word2Vec), complementados con evaluación de sesgos y explicabilidad.

---

## 👤 Autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.

---

## 🤖 Asistencia técnica

Documentación, visualizaciones y refactorización guiadas por:  
**ChatGPT (gpt-5, 2025)**
