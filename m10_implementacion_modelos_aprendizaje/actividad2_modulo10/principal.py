# principal.py
# modulo 10 - sesion 2
# contenerizacion de una api ml con docker
#
# objetivos de la sesion:
# 1) entrenar un modelo clasico y guardar el artefacto serializado.
# 2) exponer el modelo mediante una api flask con endpoints / y /predict.
# 3) preparar los archivos de despliegue: requirements.txt y dockerfile funcional.
# 4) construir y ejecutar la imagen docker mapeando el puerto 5000, validando con una prueba externa.
#
# salida: se crea la carpeta 'resultados_sesion2' junto a este archivo con artefactos y ejemplos.

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List, Tuple


# configuracion basica
CARPETA_RESULTADOS = "resultados_sesion2"
NOMBRE_MODELO = "modelo.pkl"
RANDOM_STATE = 42
N_ESTIMATORS = 300


def ruta_base() -> Path:
    # ruta del directorio donde se encuentra este archivo
    return Path(__file__).resolve().parent


def ruta_resultados() -> Path:
    # crea (si no existe) la carpeta de resultados
    carpeta = ruta_base() / CARPETA_RESULTADOS
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def entrenar_y_guardar_modelo(modelo_path: Path) -> Tuple[float, List[str], List[str]]:
    # entrena un modelo en el dataset wine y guarda el artefacto (modelo + nombres de clases y features)
    try:
        from sklearn.datasets import load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
    except ImportError as e:
        raise SystemExit(
            "faltan dependencias. instalar con: pip install scikit-learn joblib"
        ) from e

    data = load_wine()
    x, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)

    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(xte)))

    paquete = {
        "model": clf,
        "class_names": class_names,
        "feature_names": feature_names,
    }
    import joblib
    joblib.dump(paquete, modelo_path)

    return acc, class_names, feature_names


def escribir_app_py() -> str:
    # api flask con endpoints / (salud) y /predict (prediccion)
    contenido = f"""\
    # app.py
    # api ml en flask (wine + random forest)

    from flask import Flask, request, jsonify
    from pathlib import Path
    import numpy as np
    import joblib
    import os

    app = Flask(__name__)

    # se permite indicar la ruta del modelo por variable de entorno, con valor por defecto
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "{CARPETA_RESULTADOS}/{NOMBRE_MODELO}")).resolve()

    paquete = joblib.load(MODEL_PATH)
    model = paquete["model"]
    class_names = paquete.get("class_names", None)
    feature_names = paquete.get("feature_names", None)

    @app.get("/")
    def ready():
        # endpoint de salud del servicio
        info = {{
            "status": "ok",
            "message": "api disponible",
            "n_features": len(feature_names) if feature_names else None,
            "n_classes": len(class_names) if class_names else None
        }}
        return jsonify(info), 200

    @app.post("/predict")
    def predict():
        # se espera un json con 'features' = lista de {{"longitud": len(feature_names)}} numeros
        data = request.get_json(silent=True)
        if not data:
            return jsonify({{"error": "json ausente o invalido"}}), 400

        feats = data.get("features")
        if not isinstance(feats, list):
            return jsonify({{"error": "features debe ser una lista numerica"}}), 400

        if feature_names and len(feats) != len(feature_names):
            return jsonify({{"error": f"se esperaban {{len(feature_names)}} valores"}}), 400

        try:
            arr = np.array(feats, dtype=float).reshape(1, -1)
        except Exception:
            return jsonify({{"error": "features contiene valores no numericos"}}), 400

        pred_idx = int(model.predict(arr)[0])
        proba = None
        try:
            proba = model.predict_proba(arr).max().item()
        except Exception:
            pass

        label = class_names[pred_idx] if class_names and 0 <= pred_idx < len(class_names) else pred_idx
        return jsonify({{"prediction": label, "proba_max": proba}}), 200

    if __name__ == "__main__":
        # ejecutar localmente: python app.py
        # disponible en http://127.0.0.1:5000
        app.run(host="0.0.0.0", port=5000)
    """
    return textwrap.dedent(contenido)


def escribir_train_model_py() -> str:
    # script independiente de reentrenamiento (opcional)
    contenido = f"""\
    # train_model.py
    # reentrena el modelo (wine) y sobrescribe el artefacto en {CARPETA_RESULTADOS}/{NOMBRE_MODELO}

    from pathlib import Path
    import joblib
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    RANDOM_STATE = {RANDOM_STATE}
    N_ESTIMATORS = {N_ESTIMATORS}
    CARPETA_RESULTADOS = "{CARPETA_RESULTADOS}"
    NOMBRE_MODELO = "{NOMBRE_MODELO}"

    def main():
        base = Path(__file__).resolve().parent
        resultados = base / CARPETA_RESULTADOS
        resultados.mkdir(parents=True, exist_ok=True)
        modelo_path = resultados / NOMBRE_MODELO

        data = load_wine()
        x, y = data.data, data.target
        xtr, xte, ytr, yte = train_test_split(
            x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        clf.fit(xtr, ytr)
        acc = accuracy_score(yte, clf.predict(xte))

        paquete = {{
            "model": clf,
            "class_names": list(data.target_names),
            "feature_names": list(data.feature_names),
        }}
        joblib.dump(paquete, modelo_path)

        with open(resultados / "metricas_reentrenamiento.txt", "w", encoding="utf-8") as f:
            f.write(f"accuracy_test: {{acc:.4f}}\\n")

        print("modelo reentrenado. guardado en:", modelo_path)

    if __name__ == "__main__":
        main()
    """
    return textwrap.dedent(contenido)


def escribir_requirements_txt() -> str:
    # dependencias minimas para ejecutar api + entrenamiento
    return "\n".join([
        "flask>=3.0.0",
        "gunicorn>=21.2.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.0",
        "numpy>=1.26.0",
        "requests>=2.31.0",
    ]) + "\n"


def escribir_dockerfile() -> str:
    # imagen slim, instala deps y sirve la api con gunicorn
    contenido = f"""\
    # Dockerfile
    FROM python:3.11-slim

    # configuracion basica
    ENV PYTHONDONTWRITEBYTECODE=1 \\
        PYTHONUNBUFFERED=1

    # crear directorio de trabajo
    WORKDIR /app

    # instalar dependencias del sistema si fueran necesarias (comentado; descomentar si hace falta)
    # RUN apt-get update && apt-get install -y --no-install-recommends \\
    #     build-essential \\
    #  && rm -rf /var/lib/apt/lists/*

    # copiar requirements e instalar
    COPY requirements.txt /app/requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    # copiar codigo y artefactos (incluye {CARPETA_RESULTADOS}/{NOMBRE_MODELO})
    COPY . /app

    # exponer puerto
    EXPOSE 5000

    # variable de entorno con ruta del modelo (opcional)
    ENV MODEL_PATH="/app/{CARPETA_RESULTADOS}/{NOMBRE_MODELO}"

    # comando por defecto: gunicorn con 2 workers
    CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
    """
    return textwrap.dedent(contenido)


def escribir_test_api_py() -> str:
    # pruebas rapidas contra la api (local o dentro de docker)
    contenido = f"""\
    # test_api.py
    # envia solicitudes a /predict usando un ejemplo del dataset wine

    import json
    from pathlib import Path
    import requests

    def main():
        base = Path(__file__).resolve().parent
        resultados = base / "{CARPETA_RESULTADOS}"
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
    """
    return textwrap.dedent(contenido)


def escribir_ejemplos_y_docs(class_names: List[str], feature_names: List[str], acc: float) -> None:
    # guarda ejemplos json, comandos curl y un README rapido de docker
    resultados = ruta_resultados()

    # ejemplo de entrada (vector de 13 features para wine)
    ejemplo = {
        "features": [13.0, 2.3, 2.4, 16.8, 100.0, 2.8, 3.0, 0.3, 2.0, 5.0, 1.0, 3.0, 1000.0]
    }
    with open(resultados / "ejemplo_predict.json", "w", encoding="utf-8") as f:
        json.dump(ejemplo, f, ensure_ascii=False, indent=2)

    # info de clases y features
    resumen = {
        "dataset": "sklearn.datasets.load_wine",
        "clases": class_names,
        "features": feature_names,
        "nota": f"se esperan {len(feature_names)} valores en 'features', en el mismo orden que 'features'."
    }
    with open(resultados / "info_dataset.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # metricas
    with open(resultados / "metricas_entrenamiento.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy_test: {acc:.4f}\n")

    # comandos curl
    comandos = textwrap.dedent("""
    # la api debe estar arriba (python app.py) o ejecutandose en docker (ver docker_run.txt)

    # prueba de salud
    curl http://127.0.0.1:5000/

    # prueba de prediccion
    curl -X POST -H "Content-Type: application/json" \\
         -d @resultados_sesion2/ejemplo_predict.json \\
         http://127.0.0.1:5000/predict
    """).strip()
    with open(resultados / "comandos_curl.txt", "w", encoding="utf-8") as f:
        f.write(comandos + "\n")

    # guia docker
    guia_docker = textwrap.dedent("""
    docker build -t ml-api:sesion2 .
    docker run --rm -p 5000:5000 ml-api:sesion2

    # luego probar:
    python test_api.py
    # o usar curl desde comandos_curl.txt
    """).strip()
    with open(resultados / "docker_run.txt", "w", encoding="utf-8") as f:
        f.write(guia_docker + "\n")


def guardar(nombre: str, contenido: str) -> None:
    with open(ruta_base() / nombre, "w", encoding="utf-8") as f:
        f.write(contenido)


def main() -> None:
    resultados = ruta_resultados()
    modelo_path = resultados / NOMBRE_MODELO

    # entrenamiento y artefactos
    acc, class_names, feature_names = entrenar_y_guardar_modelo(modelo_path)
    escribir_ejemplos_y_docs(class_names, feature_names, acc)

    # archivos de proyecto
    guardar("app.py", escribir_app_py())
    guardar("train_model.py", escribir_train_model_py())
    guardar("requirements.txt", escribir_requirements_txt())
    guardar("Dockerfile", escribir_dockerfile())
    guardar("test_api.py", escribir_test_api_py())

    print("proyecto sesion 2 generado.")
    print(f"- carpeta '{CARPETA_RESULTADOS}' con artefactos y ejemplos.")
    print("- construir imagen: docker build -t ml-api:sesion2 .")
    print("- ejecutar contenedor: docker run --rm -p 5000:5000 ml-api:sesion2")
    print("- probar: python test_api.py (o usar curl).")


if __name__ == "__main__":
    main()