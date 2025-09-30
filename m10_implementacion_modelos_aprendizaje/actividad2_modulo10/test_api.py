# test_api.py
# envia solicitudes a /predict usando un ejemplo del dataset wine

import json
from pathlib import Path
import requests

def main():
    base = Path(__file__).resolve().parent
    resultados = base / "resultados_sesion2"
    resultados.mkdir(parents=True, exist_ok=True)

    # cargar ejemplo
    with open(resultados / "ejemplo_predict.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    url = "http://127.0.0.1:5000/predict"
    r = requests.post(url, json=payload, timeout=5)

    print("status_code:", r.status_code)
    try:
        print("respuesta:", r.json())
    except Exception:
        print("texto:", r.text[:300])

    with open(resultados / "respuesta_test_api.json", "w", encoding="utf-8") as f:
        try:
            json.dump(r.json(), f, ensure_ascii=False, indent=2)
        except Exception:
            f.write(r.text)

if __name__ == "__main__":
    main()
