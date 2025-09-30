# app.py
# api ml en flask (wine + random forest)

from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import joblib
import os

app = Flask(__name__)

# se permite indicar la ruta del modelo por variable de entorno, con valor por defecto
MODEL_PATH = Path(os.getenv("MODEL_PATH", "resultados_sesion2/modelo.pkl")).resolve()

paquete = joblib.load(MODEL_PATH)
model = paquete["model"]
class_names = paquete.get("class_names", None)
feature_names = paquete.get("feature_names", None)

@app.get("/")
def ready():
    # endpoint de salud del servicio
    info = {
        "status": "ok",
        "message": "api disponible",
        "n_features": len(feature_names) if feature_names else None,
        "n_classes": len(class_names) if class_names else None
    }
    return jsonify(info), 200

@app.post("/predict")
def predict():
    # se espera un json con 'features' = lista de {"longitud": len(feature_names)} numeros
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "json ausente o invalido"}), 400

    feats = data.get("features")
    if not isinstance(feats, list):
        return jsonify({"error": "features debe ser una lista numerica"}), 400

    if feature_names and len(feats) != len(feature_names):
        return jsonify({"error": f"se esperaban {len(feature_names)} valores"}), 400

    try:
        arr = np.array(feats, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({"error": "features contiene valores no numericos"}), 400

    pred_idx = int(model.predict(arr)[0])
    proba = None
    try:
        proba = model.predict_proba(arr).max().item()
    except Exception:
        pass

    label = class_names[pred_idx] if class_names and 0 <= pred_idx < len(class_names) else pred_idx
    return jsonify({"prediction": label, "proba_max": proba}), 200

if __name__ == "__main__":
    # ejecutar localmente: python app.py
    # disponible en http://127.0.0.1:8000
    app.run(host="0.0.0.0", port=8000)
