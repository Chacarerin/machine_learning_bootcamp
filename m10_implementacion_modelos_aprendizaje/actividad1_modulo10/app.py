# app.py
# api minima en flask para exponer el modelo de clasificacion (iris)

from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import joblib

app = Flask(__name__)

# ruta del modelo guardado
MODEL_PATH = Path(__file__).resolve().parent / "resultados_sesion1" / "modelo.pkl"

# el modelo y las etiquetas se cargan una sola vez al iniciar el servicio
paquete = joblib.load(MODEL_PATH)
model = paquete["model"]
target_names = paquete.get("target_names", None)

@app.get("/")
def ready():
    # endpoint de salud del servicio
    return jsonify({"status": "ok", "message": "api disponible"}), 200

@app.post("/predict")
def predict():
    # espera un json con clave "features" y 4 valores numericos (sepal_length, sepal_width, petal_length, petal_width)
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "json ausente o invalido"}), 400

    feats = data.get("features")
    if not isinstance(feats, list):
        return jsonify({"error": "features debe ser una lista de 4 numeros"}), 400

    if len(feats) != 4:
        return jsonify({"error": "se esperaban 4 valores para iris"}), 400

    try:
        arr = np.array(feats, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({"error": "features contiene valores no numericos"}), 400

    pred = int(model.predict(arr)[0])
    label = target_names[pred] if target_names and 0 <= pred < len(target_names) else pred

    return jsonify({"prediction": label}), 200


if __name__ == "__main__":
    # iniciar servicio local: python app.py
    # la app queda disponible en http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000)
