# 📘 actividad sesión 2 --- contenerización de una api ml con docker

este proyecto implementa la **contenerización con docker** de una api flask que expone un modelo
de clasificación entrenado con el dataset **wine**. el flujo incluye entrenamiento, serialización
del modelo, definición de dependencias (`requirements.txt`), creación de un `dockerfile` y pruebas
de la api tanto en local como dentro de un contenedor.

---

## ▶️ ejecución rápida

```bash
# generar artefactos y archivos del proyecto
python principal.py

# instalar dependencias en entorno local
pip install -r requirements.txt

# iniciar la api localmente (en puerto 8000 si se configuró así)
python app.py

# construir la imagen docker
docker build -t ml-api:sesion2 .

# ejecutar la api en contenedor
docker run --rm -p 5000:5000 ml-api:sesion2

# probar la api
python test_api.py
```

- la carpeta `resultados_sesion2/` se crea automáticamente junto al `principal.py`.  
- no se requieren datasets externos (usa `sklearn.datasets.load_wine`).  

---

## 📦 estructura del proyecto

```
actividad2_modulo10/
├── principal.py
├── app.py
├── train_model.py
├── test_api.py
├── requirements.txt
├── dockerfile
├── readme.md
└── resultados_sesion2/
    ├── modelo.pkl
    ├── metricas_entrenamiento.txt
    ├── ejemplo_predict.json
    ├── info_dataset.json
    ├── comandos_curl.txt
    ├── docker_run.txt
    └── respuesta_test_api.json
```

---

## 1) dataset y preprocesamiento

- **dataset**: wine (178 registros, 13 características, 3 clases).  
- **features**: medidas químicas del vino (ej. alcohol, ácido málico, cenizas, flavonoides, etc.).  
- **preprocesamiento**: división en entrenamiento y prueba (80/20, estratificado).  
- **modelo**: random forest con `n_estimators=300`, `random_state=42`.  

---

## 2) resultados obtenidos

- exactitud en prueba: **1.0000**  
- clases reconocidas: *class_0*, *class_1*, *class_2* (nombres originales de `sklearn.datasets.load_wine`).  
- ejemplo de entrada (`ejemplo_predict.json`):

```json
{
  "features": [13.0, 2.3, 2.4, 16.8, 100.0, 2.8, 3.0, 0.3, 2.0, 5.0, 1.0, 3.0, 1000.0]
}
```

(ver `metricas_entrenamiento.txt`, `info_dataset.json`, `ejemplo_predict.json`).  

---

## 3) análisis

- el modelo alcanzó **100% de exactitud** en el conjunto de prueba, lo que muestra su alta capacidad
  para este dataset relativamente pequeño y bien estructurado.  
- el endpoint `/predict` devuelve la clase predicha y la probabilidad asociada, lo que facilita la
  interpretación de resultados.  
- la contenerización con docker permite ejecutar la api en cualquier entorno, garantizando
  portabilidad y reproducibilidad.  
- los ejemplos json y los comandos curl permiten validar fácilmente la api tanto en local
  como dentro del contenedor.  

---

## 4) conclusión

- el objetivo de **contenerizar un modelo ml con docker** fue alcanzado.  
- se logró entrenar, serializar y exponer el modelo con flask, junto con un `dockerfile`
  funcional para despliegue portátil.  
- la exactitud perfecta refleja la simplicidad del dataset y la potencia del modelo random forest.  
- como nota práctica: basta ejecutar `python principal.py` y luego `docker build ...` para
  cumplir con lo solicitado; probar la api localmente o dentro de docker es opcional,
  pero altamente recomendable para demostrar el flujo completo.  

---

## 👤 autor

este proyecto fue desarrollado por **Rubén Schnettler**  
📍 Viña del Mar, Chile.  

---

## 🤖 asistencia técnica

documentación y apoyo en redacción por **chatgpt (gpt-5, 2025)**
