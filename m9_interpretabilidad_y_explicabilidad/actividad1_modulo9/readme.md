# 📘 actividad sesión 1 --- explicabilidad con LIME y SHAP (opiniones clínicas)

este proyecto entrena un clasificador binario **TF‑IDF + LogisticRegression** sobre un conjunto
de opiniones clínicas (positivo/negativo) y lo explica con **LIME** y **SHAP**. se generan
explicaciones para varias instancias de test y se comparan palabras destacadas por ambos métodos.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion1/`.  
- permite usar datos propios vía `--csv` (columnas `texto`, `label`) o `--txt` (`texto;label`).  
- índices a explicar configurables con `--exp_indices` (por defecto: `0,1,2`).  

---

## 📦 estructura del proyecto

```
actividad1_modulo9/
├── principal.py
├── readme.md
└── resultados_sesion1/
    ├── matriz_confusion.png
    ├── reporte_clasificacion.txt
    ├── lime_doc_0.png
    ├── lime_doc_0.html
    ├── lime_doc_1.png
    ├── lime_doc_1.html
    ├── lime_doc_2.png
    ├── lime_doc_2.html
    ├── shap_bar_doc_0.png
    ├── shap_waterfall_doc_0.png
    ├── shap_text_doc_0.html
    ├── shap_bar_doc_1.png
    ├── shap_waterfall_doc_1.png
    ├── shap_text_doc_1.html
    ├── shap_bar_doc_2.png
    ├── shap_waterfall_doc_2.png
    ├── shap_text_doc_2.html
    └── resumen.json
```

> nota: según los índices efectivamente explicados, los nombres pueden variar.

---

## 1) datos y modelo

- **dataset**: opiniones clínicas simuladas (etiquetas 0=negativo, 1=positivo).  
- **modelo**: `TF‑IDF(uni/bi‑gramas) + LogisticRegression(C=2.0, class_weight='balanced')`.  
- **split**: 70% train / 30% test con estratificación.  

---

## 2) resultados obtenidos

- **accuracy (test)** ≈ **0.80**  
- **reporte por clase** (test):  
  - **negativo** — precision: **0.6667**, recall: **1.0000**, f1: **0.8000** (soporte: 2)  
  - **positivo** — precision: **1.0000**, recall: **0.6667**, f1: **0.8000** (soporte: 3)  
- **promedios**: macro‑avg (prec/rec/f1) = **0.8333/0.8333/0.8000**, weighted‑avg f1 = **0.8000**.  

(ver `reporte_clasificacion.txt` y `matriz_confusion.png`).  

---

## 3) análisis

- la **matriz de confusión** muestra buen recuerdo para la clase **negativo** y una pérdida de recall en
  **positivo** (1 falso negativo). la precisión de **positivo** es alta, lo que sugiere umbral conservador.  
- **LIME** destaca tokens específicos que empujan la predicción; resulta útil para revisar términos que
  podrían estar sesgando el modelo (p. ej., *excelente*, *pésima*, *rápido*, *demasiado*).  
- **SHAP** ofrece una descomposición aditiva: los **bar plots** priorizan contribuciones absolutas y los
  **waterfall** detallan cómo cada término ajusta la probabilidad final. la vista **text HTML** colorea
  palabras según su impacto, ideal para auditorías rápidas.  
- en la comparación cualitativa, suele existir **intersección** entre las palabras top de LIME y SHAP; las
  discrepancias ayudan a detectar **inestabilidad local** o efectos de multicolinealidad en n‑gramas.  

---

## 4) conclusión

- se cumple el objetivo de **entrenar, evaluar y explicar** un clasificador de opiniones con **LIME** y
  **SHAP**, produciendo artefactos reproducibles y aptos para informe.  
- el desempeño (**80% de accuracy**) es coherente para un baseline TF‑IDF; las explicaciones permiten
  detectar **falsos negativos** y revisar vocabulario sensible.  
- siguientes pasos sugeridos:  
  - sintonizar **C** y **n‑gramas**, incorporar **stopwords del dominio**.  
  - usar **calibración** o ajuste de umbral para equilibrar *precision/recall* en **positivo**.  
  - comparar con modelos **transformer** (fine‑tuning ligero) y validar si las explicaciones convergen.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
