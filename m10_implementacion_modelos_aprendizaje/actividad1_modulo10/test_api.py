# test_api.py
# realiza solicitudes a /predict usando los ejemplos json generados en resultados_sesion1

import json
import time
from pathlib import Path

import requests

def main():
    base = Path(__file__).resolve().parent
    resultados = base / "resultados_sesion1"
    resultados.mkdir(parents=True, exist_ok=True)

    url = "http://127.0.0.1:5000/predict"

    ejemplos = []
    for i in range(1, 3 + 1):
        with open(resultados / f"ejemplo_{i}.json", "r", encoding="utf-8") as f:
            ejemplos.append(json.load(f))

    respuestas = []
    for idx, payload in enumerate(ejemplos, start=1):
        try:
            r = requests.post(url, json=payload, timeout=5)
            respuestas.append({
                "ejemplo": idx,
                "payload": payload,
                "status_code": r.status_code,
                "respuesta": intentar_json(r)
            })
            time.sleep(0.4)
        except requests.exceptions.RequestException as e:
            respuestas.append({
                "ejemplo": idx,
                "payload": payload,
                "status_code": None,
                "error": str(e)
            })

    with open(resultados / "pruebas_respuestas.json", "w", encoding="utf-8") as f:
        json.dump(respuestas, f, ensure_ascii=False, indent=2)

    with open(resultados / "log_pruebas.txt", "w", encoding="utf-8") as f:
        for r in respuestas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("pruebas finalizadas. revisar 'pruebas_respuestas.json' y 'log_pruebas.txt'.")

def intentar_json(resp):
    try:
        return resp.json()
    except Exception:
        # si no es json valido, se guarda un extracto del texto
        return {"contenido": resp.text[:300]}

if __name__ == "__main__":
    main()
