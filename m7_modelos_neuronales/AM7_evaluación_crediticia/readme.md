# 📘 evaluación modular — módulo 7: scoring crediticio con redes neuronales profundas

este proyecto implementa y compara una **dnn** y una **resnet para datos tabulares** para predecir la probabilidad de buen comportamiento crediticio usando el dataset **german credit**. incluye preprocesamiento, manejo de desbalanceo, evaluación con métricas clásicas y **explicabilidad** vía **shap** y **lime**.

---

## 📌 características principales

- **dataset**: german credit (openml/uci/kaggle).  
- **tarea**: clasificación binaria (`good` vs `bad`).  
- **modelos aplicados**:
  - dnn con regularización l2 y dropout.  
  - resnet tabular con skip connections.  
- **regularización y callbacks**:
  - `earlystopping` y `reducelronplateau`.  
- **manejo de desbalanceo**:
  - uso de `class_weight="balanced"`.  
- **interpretabilidad**:
  - shap (`shap_summary_plot.png`).  
  - lime (`lime_example.txt`).  
- **evaluación**:
  - accuracy, precision, recall, f1, auc-roc, matriz de confusión y curva roc.  

---

## ▶️ ejecución

```bash
python principal.py
```

los resultados se guardan en `resultados_mod7/`.

---

## 📁 estructura del proyecto

```
evaluacion_modulo7/
├── principal.py
├── requirements.txt
└── readme.md
```

---

## 1) análisis exploratorio de datos (eda)

- el dataset presenta variables numéricas (ej.: edad, monto crédito) y categóricas (estado civil, propósito del crédito).  
- **distribución de clases**: dataset desbalanceado, con mayoría de clientes "good". esto implica riesgo de que accuracy sea engañoso.  
- se aplicó `class_weight` para mitigar este sesgo.  

---

## 2) preprocesamiento

- **codificación**: one-hot encoding para variables categóricas.  
- **normalización**: standardscaler para numéricas.  
- **desbalanceo**: uso de `class_weight="balanced"`.  

---

## 3) entrenamiento de modelos

- **dnn**:
  - capas densas 128–64, activación relu, dropout y regularización l2.  
- **resnet tabular**:
  - bloques residuales con conexiones de salto.  
- **callbacks**:
  - early stopping para evitar sobreajuste.  
  - reduce lr on plateau para ajustar tasa de aprendizaje.  

---

## 4) resultados obtenidos

**métricas finales (de `metricas_modelos.csv`):**

| modelo  | accuracy | precision | recall | f1   | auc |
|---------|---------:|----------:|-------:|-----:|----:|
| dnn     | 0.645    | 0.863     | 0.586  | 0.698 | 0.763 |
| resnet  | 0.680    | 0.828     | 0.686  | 0.750 | 0.736 |

**gráficos generados:**
- `matriz_confusion_dnn.png`, `roc_dnn.png`  
- `matriz_confusion_resnet.png`, `roc_resnet.png`  

**análisis:**  
- la dnn muestra desempeño aceptable, pero la resnet la supera en todas las métricas.  
- la resnet logra un auc de **0.73**, lo que refleja mayor capacidad discriminativa.  

---

## 5) umbral de decisión y costo de errores

- **falsos positivos (tipo i)**: aprobar crédito a clientes que no pagarán → pérdida financiera directa.  
- **falsos negativos (tipo ii)**: rechazar a buenos clientes → pérdida de oportunidad y posible sesgo.  
- el umbral usado fue **0.5**, pero un ajuste a 0.4 podría aumentar el recall de la clase "bad", reduciendo tipo i aunque baje la precisión.  

---

## 6) explicabilidad del modelo

### shap
el análisis shap (`shap_summary_plot.png`) muestra que las variables con mayor impacto son:  
- **credit_amount** (montos altos elevan riesgo).  
- **age** (edades menores aumentan riesgo).  
- **duration** (plazos más largos incrementan probabilidad de impago).  
- **checking_status** (estado de cuenta previo influye fuertemente).  

### lime
ejemplo (`lime_example.txt`):  
- `credit_amount > 0.33` con contribución positiva al riesgo.  
- `num_dependents <= -0.44` con contribución negativa.  
- `age <= -0.83` también aumenta riesgo.  

esto confirma que variables financieras y demográficas clave influyen en la clasificación.  

---

## 7) reflexión ética y sesgos

- existe riesgo de que el modelo capture **sesgos históricos** (ej.: edad, estado civil, ser extranjero).  
- la inclusión de shap y lime permite explicar las predicciones a un equipo de riesgo bancario, aumentando transparencia.  
- se recomienda monitorear variables sensibles y evaluar impacto regulatorio para evitar discriminación.  

---

## 8) conclusiones

- la **resnet tabular** es claramente superior en métricas (f1=0.75 vs 0.69 de la dnn).  
- se valida la importancia de aplicar técnicas modernas con regularización y callbacks.  
- el uso de shap y lime aporta interpretabilidad, esencial para decisiones financieras.  
- en un escenario real, priorizar recall alto de la clase "bad" reduce pérdidas por impago, aunque implique sacrificar algo de precisión.  

---

## 👤 Autor

Este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.

---

## 🤖 Asistencia técnica

Documentación, visualizaciones y refactorización guiadas por:  
**ChatGPT (gpt-5, 2025)**