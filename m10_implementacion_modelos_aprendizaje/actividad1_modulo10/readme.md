# 📘 actividad sesión 1 --- despliegue básico de un modelo de clasificación con flask

este proyecto implementa un **flujo mínimo de despliegue** para un modelo de clasificación usando el dataset **iris**.  
se entrena un modelo **random forest** y se expone mediante una api **flask** con un endpoint `/predict`.  
el proyecto incluye scripts auxiliares, artefactos de entrenamiento, ejemplos de prueba y guías de uso.

---

## ▶️ ejecución rápida

```bash
python principal.py
```

- genera todas las salidas en `resultados_sesion1/`.  
- no requiere datasets externos (usa `sklearn.datasets.load_iris`).  
- la api se inicia con `python app.py` y se prueba con `python test_api.py` o `curl`.  

---

## 📦 estructura del proyecto

```
actividad1_modulo10/
├── principal.py
├── app.py
├── entrenar_modelo.py
├── test_api.py
├── readme.md
└── resultados_sesion1/
    ├── modelo.pkl
    ├── metricas_entrenamiento.txt
    ├── ejemplo_1.json
    ├── ejemplo_2.json
    ├── ejemplo_3.json
    ├── info_clases.json
    ├── comandos_curl.txt
    ├── pruebas_respuestas.json
    ├── log_pruebas.txt
    └── guia_rapida.txt
```

---

## 1) dataset y preprocesamiento

- **dataset**: iris (150 registros, 3 clases: *setosa*, *versicolor*, *virginica*).  
- **features**: 4 variables numéricas (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).  
- **preprocesamiento**: división en entrenamiento y prueba (80/20, estratificado).  
- **modelo**: random forest con `n_estimators=200`, `random_state=42`.  

---

## 2) resultados obtenidos

- exactitud en prueba: **0.9000**  
- las clases reconocidas son: *setosa*, *versicolor*, *virginica*.  
- ejemplos de entrada generados en formato json:  
  - `ejemplo_1.json`: `[5.1, 3.5, 1.4, 0.2]`  
  - `ejemplo_2.json`: `[6.0, 2.9, 4.5, 1.5]`  
  - `ejemplo_3.json`: `[6.9, 3.1, 5.4, 2.1]`  

(ver `metricas_entrenamiento.txt`, `info_clases.json`, `ejemplo_*.json`).  

---

## 3) análisis

- el modelo logra **buena exactitud (90%)** en un dataset estándar, lo que valida su uso como demostración.  
- el endpoint `/predict` acepta un json con la clave `"features"` y devuelve la clase predicha en texto.  
- la api responde de manera robusta a entradas válidas, y entrega mensajes claros de error cuando faltan valores o el json no es correcto.  
- la generación de ejemplos y comandos curl facilita la reproducibilidad de las pruebas.  

---

## 4) conclusión

- el objetivo de **desplegar un modelo clásico en flask** fue alcanzado.  
- se logró entrenar, serializar y exponer el modelo, junto con ejemplos de prueba listos para usar.  
- la actividad demuestra un flujo reproducible que puede ampliarse fácilmente para modelos más complejos.  
- como nota práctica: **ejecutar únicamente `python principal.py` ya cumple con lo solicitado**, pues genera el modelo y los artefactos.  
  levantar la api (`app.py`) y probar (`test_api.py` o `curl`) es opcional, pero recomendable para mostrar el funcionamiento en vivo.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
