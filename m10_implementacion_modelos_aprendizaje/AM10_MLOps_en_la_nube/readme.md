# 📘 actividad módulo 10 — mlops en la nube (despliegue automatizado de un modelo predictivo)

este proyecto implementa un flujo completo de **mlops local**:  
1. entrenamiento y serialización de un modelo predictivo (**regresión logística** sobre el dataset *breast cancer wisconsin*).  
2. exposición del modelo como **api rest con flask** (`/` y `/predict`).  
3. **dockerización** del sistema para asegurar reproducibilidad.  

---

## ▶️ ejecución rápida

### entrenar el modelo
```bash
python modelo.py
```
- genera `resultados/modelo.joblib` y `resultados/metadata.json`.  
- guarda métricas de validación (accuracy y roc-auc) en consola.  

### levantar la api local
```bash
python app.py
# o recomendado en producción:
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

navegar a [http://127.0.0.1:8000/](http://127.0.0.1:8000/) para comprobar estado.  

### docker
```bash
docker build -t modulo10_api .
docker run --rm -p 8000:8000 modulo10_api
```

---

## 📦 estructura del proyecto

```
actividad_modulo10/
├── app.py
├── modelo.py
├── requirements.txt
├── Dockerfile
├── readme.md
├── resultados/
│   ├── modelo.joblib
│   └── metadata.json
├── tests/
│   ├── test_app.py
│   └── prueba_api.py
├── capturas_pantalla/
│   ├── captura01.jpg
│   └── captura02.jpg
│   └── captura03.jpg

```

---

## 🧪 pruebas realizadas

### 1. endpoint `/`
al acceder a `http://127.0.0.1:8000/` se obtuvo:

```json
{
  "status": "ok",
  "mensaje": "servicio de predicción activo",
  "modelo": "pipeline(StandardScaler + LogisticRegression)",
  "features": ["mean radius", "mean texture", "..."]
}
```

esto demuestra que el servicio flask está arriba y cargó correctamente el modelo.  

---

### 2. endpoint `/predict` con curl
```bash
curl -s -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"data": {"mean radius": 14.2, "mean texture": 20.1, ... }}'
```

respuesta ejemplo:
```json
{
  "ok": true,
  "n": 1,
  "resultados": [
    {
      "prediccion": 1,
      "probabilidad_clase_1": 0.989,
      "clases": ["malignant", "benign"],
      "nota": "por convención sklearn, la clase 1 suele corresponder a 'malignant'"
    }
  ]
}
```

---

### 3. prueba con python (`prueba_api.py`)
se implementó un script cliente que primero consulta `/` y luego envía un registro al endpoint `/predict`.  
ejemplo de salida real en consola:

```text
200
{'n': 1, 'ok': True, 'resultados': [{'clases': ['malignant', 'benign'], 
 'nota': "por convención sklearn, la clase 1 suele corresponder a 'malignant'", 
 'prediccion': 1, 'probabilidad_clase_1': 0.989094072452638}]}
```

esto valida que la api funciona correctamente y devuelve predicciones confiables.  

---

## 📌 conclusiones

- se logró entrenar y serializar un modelo predictivo con métricas aceptables.  
- la api flask expone correctamente el modelo y valida entradas.  
- se realizaron pruebas locales exitosas tanto con curl, navegador como con python.  
- la dockerización permite levantar el servicio de manera reproducible en cualquier entorno.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
