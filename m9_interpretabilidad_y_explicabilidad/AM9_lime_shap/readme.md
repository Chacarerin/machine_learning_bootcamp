# ❤️ evaluación modular — módulo 9: interpretabilidad de modelos predictivos con lime y shap

este proyecto implementa un sistema de clasificación para predecir **enfermedad cardíaca** utilizando el dataset público de kaggle *heart failure prediction*. se construye un modelo **random forest**, se aplican técnicas de **explicabilidad (shap y lime)**, y se analizan posibles **sesgos por variables sensibles**.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- si no existe `heart.csv` local, se descarga automáticamente desde kaggle con `kagglehub`.  
- los resultados se guardan en la carpeta `resultados_mod9/` junto al script.

---

## 📦 estructura

```
actividad_modulo9/
├── principal.py
├── heart.csv  (opcional, puede descargarse automáticamente)
└── readme.md
```

---

## 1) datos y eda

- **filas**: 918  
- **columnas**: 12  
- **target**: `heartdisease` (0 = no, 1 = sí)  

principales variables: `age`, `sex`, `chestpaintype`, `restingbp`, `cholesterol`, `fastingbs`, `restingecg`, `maxhr`, `exerciseangina`, `oldpeak`, `st_slope`.  

> variables sensibles: `sex`, además de posibles umbrales etarios derivados de `age`.  

---

## 2) preprocesamiento

- escalado de variables numéricas (`standardscaler`).  
- codificación one-hot para variables categóricas (`sex`, `chestpaintype`, `restingecg`, `exerciseangina`, `st_slope`).  
- división **train/test = 75% / 25%**, estratificada en la variable objetivo.  

---

## 3) modelado

- modelo principal: **random forest classifier** (`n_estimators=400`).  
- se evaluaron métricas globales: accuracy, precision, recall, f1, auc.  

---

## 4) resultados obtenidos

**métricas globales (random forest):**

| métrica    | valor |
|------------|-------|
| accuracy   | 0.904 |
| precision  | 0.895 |
| recall     | 0.937 |
| f1-score   | 0.915 |
| auc        | 0.944 |

> el modelo logró un **f1≈0.92** y **auc≈0.94**, mostrando buen equilibrio entre precisión y sensibilidad.

**gráficos generados:**  
- `matriz_confusion_rf.png`  
- `roc_rf.png`  
- `shap_summary_dot.png`  
- `shap_summary_bar.png`

---

## 5) explicabilidad con shap

- **gráfico summary (dot)**: muestra que variables como `st_slope`, `chestpaintype` y `oldpeak` tienen fuerte impacto en la predicción.  
- **summary bar**: confirma que `st_slope` y `chestpaintype` lideran la importancia global.  
- **casos locales (waterfall/top10 contribuciones)**: reflejan cómo combinaciones de factores clínicos individuales influyen en la predicción de enfermedad.  

---

## 6) explicabilidad con lime

se analizaron los **mismos 3 casos** que en shap. principales observaciones:

- **caso 1** (`lime_case_1.txt`): `st_slope_Up`, `chestpaintype_ASY` y `st_slope_Flat` favorecen la predicción positiva; el género femenino resta peso.  
- **caso 2** (`lime_case_2.txt`): importancia positiva de `st_slope_Up` y `chestpaintype_ASY`; valores bajos de colesterol también reducen el riesgo.  
- **caso 3** (`lime_case_3.txt`): `cholesterol` y `maxhr` aportan hacia predicción positiva, mientras que `st_slope_Up` y `chestpaintype_ASY` reducen el riesgo.  

> comparando shap y lime: ambos resaltan la influencia de `st_slope` y `chestpaintype`, aunque lime enfatiza además interacciones locales específicas.

---

## 7) sesgos y equidad

archivo `sesgo_subgrupos.csv`:

- **sexo**: desempeño consistente entre hombres y mujeres, sin diferencias críticas en f1.  
- **edad**: métricas relativamente estables en los grupos `<=39`, `40-54` y `55+`.  

> no se detectan sesgos fuertes, aunque se recomienda monitoreo en aplicaciones reales.

---

## 8) reflexión ética

- **riesgos**: usar el modelo sin interpretabilidad podría llevar a decisiones médicas poco transparentes.  
- **medidas**: auditorías regulares, monitoreo de sesgos, revisión médica de predicciones críticas.  
- **responsabilidad**: el sistema debe servir como **apoyo clínico**, nunca como reemplazo de la evaluación profesional.

---

## 9) conclusiones

- el random forest logró métricas sólidas (auc≈0.94).  
- shap y lime muestran consistencia en la importancia de `st_slope` y `chestpaintype`.  
- no se observaron sesgos críticos por sexo o edad, pero se enfatiza la importancia de vigilancia continua.  
- se valida la necesidad de **explicabilidad + análisis ético** en modelos de salud.

---

## 👤 autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del mar, Chile.

---

## 🤖 asistencia técnica

documentación, visualizaciones y refactorización guiadas por:  
**chatgpt (gpt-5, 2025)**
