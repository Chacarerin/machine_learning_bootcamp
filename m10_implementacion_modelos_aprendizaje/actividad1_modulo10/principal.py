# principal.py
# modulo 10 - sesion 1
# despliegue basico de un modelo de clasificacion con flask
#
# objetivos de la sesion:
# 1) preparar un flujo minimo de despliegue para un modelo clasico (iris) usando flask.
# 2) asegurar reproducibilidad: crear una carpeta de resultados junto al script con artefactos y logs.
# 3) generar archivos auxiliares: api (app.py), script de reentrenamiento (entrenar_modelo.py),
#    script de pruebas (test_api.py) y ejemplos json para el endpoint /predict.
# 4) documentar el proceso con una guia rapida (guia_rapida.txt).


from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List, Tuple


# configuracion simple del proyecto
NOMBRE_CARPETA_RESULTADOS = "resultados_sesion1"
NOMBRE_MODELO = "modelo.pkl"
RANDOM_STATE = 42
N_ESTIMATORS = 200


def ruta_base() -> Path:
    # retorna la carpeta fisica donde se encuentra este archivo principal.py
    return Path(__file__).resolve().parent


def ruta_resultados() -> Path:
    # crea (si no existe) la carpeta de resultados junto al archivo principal.py
    carpeta = ruta_base() / NOMBRE_CARPETA_RESULTADOS
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def entrenar_y_guardar_modelo(modelo_path: Path) -> Tuple[float, List[str]]:
    # entrena un clasificador simple en iris y guarda el modelo serializado
    # ademas calcula una exactitud de referencia sobre un conjunto de prueba
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
    except ImportError as e:
        raise SystemExit(
            "falta instalar dependencias. ejecutar: pip install scikit-learn joblib"
        ) from e

    iris = load_iris()
    x, y = iris.data, iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)

    acc = float(accuracy_score(y_test, clf.predict(x_test)))

    # se incluye el nombre de clases para que la api devuelva etiqueta legible
    paquete = {"model": clf, "target_names": iris.target_names.tolist()}

    import joblib
    joblib.dump(paquete, modelo_path)

    return acc, iris.target_names.tolist()


def escribir_app_py() -> str:
    # genera el contenido del archivo app.py con un endpoint /predict
    # se asume que el modelo fue guardado previamente en la carpeta de resultados
    contenido = f"""\
    # app.py
    # api minima en flask para exponer el modelo de clasificacion (iris)

    from flask import Flask, request, jsonify
    from pathlib import Path
    import numpy as np
    import joblib

    app = Flask(__name__)

    # ruta del modelo guardado
    MODEL_PATH = Path(__file__).resolve().parent / "{NOMBRE_CARPETA_RESULTADOS}" / "{NOMBRE_MODELO}"

    # el modelo y las etiquetas se cargan una sola vez al iniciar el servicio
    paquete = joblib.load(MODEL_PATH)
    model = paquete["model"]
    target_names = paquete.get("target_names", None)

    @app.get("/")
    def ready():
        # endpoint de salud del servicio
        return jsonify({{"status": "ok", "message": "api disponible"}}), 200

    @app.post("/predict")
    def predict():
        # espera un json con clave "features" y 4 valores numericos (sepal_length, sepal_width, petal_length, petal_width)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({{"error": "json ausente o invalido"}}), 400

        feats = data.get("features")
        if not isinstance(feats, list):
            return jsonify({{"error": "features debe ser una lista de 4 numeros"}}), 400

        if len(feats) != 4:
            return jsonify({{"error": "se esperaban 4 valores para iris"}}), 400

        try:
            arr = np.array(feats, dtype=float).reshape(1, -1)
        except Exception:
            return jsonify({{"error": "features contiene valores no numericos"}}), 400

        pred = int(model.predict(arr)[0])
        label = target_names[pred] if target_names and 0 <= pred < len(target_names) else pred

        return jsonify({{"prediction": label}}), 200


    if __name__ == "__main__":
        # iniciar servicio local: python app.py
        # la app queda disponible en http://127.0.0.1:5000
        app.run(host="0.0.0.0", port=5000)
    """
    return textwrap.dedent(contenido)


def escribir_entrenar_modelo_py() -> str:
    # genera un script independiente para repetir el entrenamiento cuando sea necesario
    contenido = f"""\
    # entrenar_modelo.py
    # reentrena el modelo iris y sobrescribe el archivo de artefacto en {NOMBRE_CARPETA_RESULTADOS}/{NOMBRE_MODELO}

    from pathlib import Path
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    RANDOM_STATE = {RANDOM_STATE}
    N_ESTIMATORS = {N_ESTIMATORS}
    CARPETA_RESULTADOS = "{NOMBRE_CARPETA_RESULTADOS}"
    NOMBRE_MODELO = "{NOMBRE_MODELO}"

    def main():
        base = Path(__file__).resolve().parent
        resultados = base / CARPETA_RESULTADOS
        resultados.mkdir(parents=True, exist_ok=True)

        iris = load_iris()
        x, y = iris.data, iris.target
        xtr, xte, ytr, yte = train_test_split(
            x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        clf.fit(xtr, ytr)

        joblib.dump({{"model": clf, "target_names": iris.target_names.tolist()}}, resultados / NOMBRE_MODELO)

        acc = accuracy_score(yte, clf.predict(xte))
        with open(resultados / "metricas_reentrenamiento.txt", "w", encoding="utf-8") as f:
            f.write(f"accuracy_test: {{acc:.4f}}\\n")

        print("modelo reentrenado y guardado en:", resultados / NOMBRE_MODELO)

    if __name__ == "__main__":
        main()
    """
    return textwrap.dedent(contenido)


def escribir_test_api_py() -> str:
    # genera un script para pruebas rapidas contra la api local
    contenido = f"""\
    # test_api.py
    # realiza solicitudes a /predict usando los ejemplos json generados en {NOMBRE_CARPETA_RESULTADOS}

    import json
    import time
    from pathlib import Path

    import requests

    def main():
        base = Path(__file__).resolve().parent
        resultados = base / "{NOMBRE_CARPETA_RESULTADOS}"
        resultados.mkdir(parents=True, exist_ok=True)

        url = "http://127.0.0.1:5000/predict"

        ejemplos = []
        for i in range(1, 3 + 1):
            with open(resultados / f"ejemplo_{{i}}.json", "r", encoding="utf-8") as f:
                ejemplos.append(json.load(f))

        respuestas = []
        for idx, payload in enumerate(ejemplos, start=1):
            try:
                r = requests.post(url, json=payload, timeout=5)
                respuestas.append({{
                    "ejemplo": idx,
                    "payload": payload,
                    "status_code": r.status_code,
                    "respuesta": intentar_json(r)
                }})
                time.sleep(0.4)
            except requests.exceptions.RequestException as e:
                respuestas.append({{
                    "ejemplo": idx,
                    "payload": payload,
                    "status_code": None,
                    "error": str(e)
                }})

        with open(resultados / "pruebas_respuestas.json", "w", encoding="utf-8") as f:
            json.dump(respuestas, f, ensure_ascii=False, indent=2)

        with open(resultados / "log_pruebas.txt", "w", encoding="utf-8") as f:
            for r in respuestas:
                f.write(json.dumps(r, ensure_ascii=False) + "\\n")

        print("pruebas finalizadas. revisar 'pruebas_respuestas.json' y 'log_pruebas.txt'.")

    def intentar_json(resp):
        try:
            return resp.json()
        except Exception:
            # si no es json valido, se guarda un extracto del texto
            return {{"contenido": resp.text[:300]}}

    if __name__ == "__main__":
        main()
    """
    return textwrap.dedent(contenido)


def escribir_ejemplos_json(target_names: List[str]) -> None:
    # crea tres entradas de prueba para /predict y un archivo con comandos curl
    resultados = ruta_resultados()

    ejemplos = [
        {"features": [5.1, 3.5, 1.4, 0.2]},   # setosa aproximado
        {"features": [6.0, 2.9, 4.5, 1.5]},   # versicolor aproximado
        {"features": [6.9, 3.1, 5.4, 2.1]},   # virginica aproximado
    ]
    for i, ej in enumerate(ejemplos, start=1):
        with open(resultados / f"ejemplo_{i}.json", "w", encoding="utf-8") as f:
            json.dump(ej, f, ensure_ascii=False, indent=2)

    comandos = textwrap.dedent(f"""
    # comandos de prueba (la api debe estar activa con: python app.py)
    curl -X POST -H "Content-Type: application/json" -d @{NOMBRE_CARPETA_RESULTADOS}/ejemplo_1.json http://127.0.0.1:5000/predict
    curl -X POST -H "Content-Type: application/json" -d @{NOMBRE_CARPETA_RESULTADOS}/ejemplo_2.json http://127.0.0.1:5000/predict
    curl -X POST -H "Content-Type: application/json" -d @{NOMBRE_CARPETA_RESULTADOS}/ejemplo_3.json http://127.0.0.1:5000/predict
    """).strip()
    with open(resultados / "comandos_curl.txt", "w", encoding="utf-8") as f:
        f.write(comandos + "\n")

    resumen = {
        "dataset": "iris",
        "clases": target_names,
        "nota": "los valores de 'features' son [sepal_length, sepal_width, petal_length, petal_width].",
    }
    with open(resultados / "info_clases.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)


def escribir_metricas_txt(acc: float) -> None:
    # guarda una linea con la exactitud alcanzada durante el entrenamiento rapido
    resultados = ruta_resultados()
    with open(resultados / "metricas_entrenamiento.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy_test_interno: {acc:.4f}\n")


def guardar_archivo(nombre: str, contenido: str) -> None:
    # escribe un archivo de texto en la misma carpeta de principal.py
    with open(ruta_base() / nombre, "w", encoding="utf-8") as f:
        f.write(contenido)


def escribir_guia_rapida() -> None:
    # crea una guia minima de ejecucion dentro de la carpeta de resultados
    resultados = ruta_resultados()
    guia = textwrap.dedent(f"""
    guia rapida de ejecucion
    1) crear entorno e instalar dependencias:
       python -m venv .venv
       source .venv/bin/activate  # mac/linux
       # .venv\\Scripts\\activate  # windows
       pip install flask scikit-learn joblib requests

    2) generar artefactos y archivos del proyecto:
       python principal.py

    3) levantar la api:
       python app.py
       # disponible en http://127.0.0.1:5000

    4) probar el endpoint /predict en otra terminal:
       python test_api.py
       # o usar los comandos en {NOMBRE_CARPETA_RESULTADOS}/comandos_curl.txt

    artefactos principales:
       - {NOMBRE_CARPETA_RESULTADOS}/{NOMBRE_MODELO}
       - {NOMBRE_CARPETA_RESULTADOS}/metricas_entrenamiento.txt
       - {NOMBRE_CARPETA_RESULTADOS}/ejemplo_1.json, ejemplo_2.json, ejemplo_3.json
       - {NOMBRE_CARPETA_RESULTADOS}/pruebas_respuestas.json
       - {NOMBRE_CARPETA_RESULTADOS}/log_pruebas.txt
       - {NOMBRE_CARPETA_RESULTADOS}/info_clases.json
       - {NOMBRE_CARPETA_RESULTADOS}/comandos_curl.txt
    """).strip()

    with open(resultados / "guia_rapida.txt", "w", encoding="utf-8") as f:
        f.write(guia + "\n")


def main() -> None:
    # orquesta el flujo completo de la sesion: carpeta de resultados, entrenamiento,
    # generacion de archivos auxiliares y material de apoyo
    resultados = ruta_resultados()
    modelo_path = resultados / NOMBRE_MODELO

    acc, target_names = entrenar_y_guardar_modelo(modelo_path)
    escribir_metricas_txt(acc)

    guardar_archivo("app.py", escribir_app_py())
    guardar_archivo("entrenar_modelo.py", escribir_entrenar_modelo_py())
    guardar_archivo("test_api.py", escribir_test_api_py())

    escribir_ejemplos_json(target_names)
    escribir_guia_rapida()

    print("proyecto generado.")
    print(f"- carpeta '{NOMBRE_CARPETA_RESULTADOS}' creada con artefactos y guias.")
    print("- iniciar api con: python app.py")
    print("- ejecutar pruebas con: python test_api.py")


if __name__ == "__main__":
    main()