# 📘 actividad sesión 2 --- explicabilidad local con LIME (opiniones clínicas)

este proyecto entrena un clasificador binario **TF‑IDF + LogisticRegression** y explica sus predicciones
con **LIME** sobre un conjunto breve de opiniones clínicas (0=negativo, 1=positivo). se generan
artefactos gráficos (PNG/HTML) para ≥3 ejemplos del set de test.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion2/`.  
- permite usar datos propios: `--csv opiniones.csv --col_texto texto --col_label label` o `--txt opiniones.txt`.  
- índices a explicar configurables con `--exp_indices` (por defecto: `0,1,2`).  

---

## 📦 estructura del proyecto

```
actividad2_modulo9/
├── principal.py
├── readme.md
└── resultados_sesion2/
    ├── matriz_confusion.png
    ├── reporte_clasificacion.txt
    ├── lime_doc_0.png
    ├── lime_doc_0.html
    ├── lime_doc_1.png
    ├── lime_doc_1.html
    ├── lime_doc_2.png
    ├── lime_doc_2.html
    └── resumen.json
```
> los nombres `lime_doc_*` dependen de los índices efectivamente explicados.

---

## 1) datos y modelo

- **n_total** = **15** | **train/test** = **10/5** (estratificado).  
- **modelo**: `TF-IDF(1..2) + LogisticRegression(C=2.0)`.  
- **LIME**: top **10** términos por explicación.  
- **ejemplos explicados** (test): [0, 1, 2].  

---

## 2) resultados obtenidos

- **accuracy (test)** = **0.8000**  
- **reporte por clase (test)**: ver `reporte_clasificacion.txt`.  
  - negativo — precision 0.6667, recall 1.0000, f1 0.8000 (soporte 2)  
  - positivo — precision 1.0000, recall 0.6667, f1 0.8000 (soporte 3)  
- **matriz de confusión**: ver `matriz_confusion.png` (2 TN, 2 TP, 1 FN).  

---

## 3) análisis

- el modelo base **TF‑IDF + LR** logra un desempeño **estable** en un set pequeño (accuracy ≈0.80).  
  la **clase positivo** muestra menor *recall* (un falso negativo), mientras que **negativo** se recupera bien.  
- **LIME** evidencia qué tokens empujan la predicción en cada instancia. esto permite:  
  1) detectar términos “gatillo” (p. ej., *excelente*, *pésima*, *rápido*, *demasiado*) que pueden sesgar,  
  2) depurar el preprocesamiento (stopwords del dominio, normalización de tildes/variantes), y  
  3) diseñar **reglas de negocio** (alertas cuando una palabra clave aparezca con alta contribución).  
- las explicaciones en **HTML** son útiles para revisión manual: colorean palabras por contribución y
  facilitan la discusión con perfiles no técnicos.  

---

## 4) conclusión

- se cumple el objetivo: **entrenar**, **evaluar** y **explicar localmente** con **LIME** al menos 3
  ejemplos de test, dejando artefactos reproducibles.  
- el análisis sugiere dos líneas de mejora:  
  - sintonizar **C** y ampliar n‑gramas (**tri‑gramas**) para capturar contexto;  
  - enriquecer el pipeline con **stopwords médicas** y lematización, apuntando a aumentar el *recall* en positivo.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
