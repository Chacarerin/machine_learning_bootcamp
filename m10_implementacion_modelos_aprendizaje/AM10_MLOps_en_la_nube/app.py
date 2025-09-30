
# api flask para servir el modelo entrenado
# rutas:
#   GET  /          -> estado del servicio
#   POST /predict   -> recibe json con features y retorna predicción
#
# forma de entrada para /predict:
# {
#   "data": {
#       "mean radius": 14.2,
#       "mean texture": 20.1,
#       ...
#   }
# }
#
# también se acepta una lista de registros:
# {
#   "data": [
#       {"mean radius": 14.2, "mean texture": 20.1, ...},
#       {"mean radius": 12.5, "mean texture": 18.3, ...}
#   ]
# }

import json
import joblib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Union

from flask import Flask, request, jsonify

# configuración simple de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("api")

# cargo modelo y metadata al iniciar
RUTA_MODELO = Path("resultados/modelo.joblib")
RUTA_META = Path("resultados/metadata.json")

if not RUTA_MODELO.exists() or not RUTA_META.exists():
    raise RuntimeError("no se encuentran 'resultados/modelo.joblib' o 'resultados/metadata.json'. "
                       "primero ejecuta: python modelo.py")

MODELO = joblib.load(RUTA_MODELO)
METADATA = json.loads(RUTA_META.read_text(encoding="utf-8"))
FEATURES_ESPERADAS: List[str] = METADATA["feature_names"]

app = Flask(__name__)

def validar_entrada(payload: dict) -> Tuple[List[Dict[str, float]], List[str]]:
    """valida y normaliza la entrada. retorna lista de registros dict y lista de errores."""
    errores = []
    if "data" not in payload:
        return [], ["falta la clave 'data' en el json"]

    data = payload["data"]
    registros: List[Dict[str, float]] = []

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or len(data) == 0:
        return [], ["'data' debe ser un objeto o una lista no vacía"]

    for i, fila in enumerate(data):
        if not isinstance(fila, dict):
            errores.append(f"fila {i}: el elemento debe ser un objeto con pares clave-valor")
            continue

        faltantes = [c for c in FEATURES_ESPERADAS if c not in fila]
        if faltantes:
            errores.append(f"fila {i}: faltan columnas {faltantes}")
            continue

        # convierto a float y ordeno por features esperadas
        registro = {}
        try:
            for c in FEATURES_ESPERADAS:
                registro[c] = float(fila[c])
        except (ValueError, TypeError):
            errores.append(f"fila {i}: valores no numéricos detectados")
            continue

        registros.append(registro)

    return registros, errores

@app.get("/")
def raiz():
    return jsonify({
        "status": "ok",
        "mensaje": "servicio de predicción activo",
        "modelo": METADATA.get("modelo"),
        "features": FEATURES_ESPERADAS
    })

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.exception("error parseando json")
        return jsonify({"error": "json inválido", "detalle": str(e)}), 400

    registros, errores = validar_entrada(payload)
    if errores:
        return jsonify({"error": "validación de entrada", "detalles": errores}), 400

    # preparo matriz en el orden de las features esperadas
    import numpy as np
    X = np.array([[r[c] for c in FEATURES_ESPERADAS] for r in registros])

    try:
        probas = MODELO.predict_proba(X)[:, 1].tolist()
        preds = MODELO.predict(X).tolist()
    except Exception as e:
        logger.exception("error durante la predicción")
        return jsonify({"error": "fallo al predecir", "detalle": str(e)}), 500

    # armo respuesta fila a fila
    resultados = []
    for pred, prob in zip(preds, probas):
        resultados.append({
            "prediccion": int(pred),
            "probabilidad_clase_1": float(prob),
            "clases": METADATA.get("target_names", ["negativo", "positivo"]),
            "nota": "por convención sklearn, la clase 1 suele corresponder a 'malignant'"
        })

    return jsonify({
        "ok": True,
        "n": len(resultados),
        "resultados": resultados
    })

if __name__ == "__main__":
    # modo desarrollo: flask built-in
    # producción: usar gunicorn (ver dockerfile)
    app.run(host="0.0.0.0", port=8000)
