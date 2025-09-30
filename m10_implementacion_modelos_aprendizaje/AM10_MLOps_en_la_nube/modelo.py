# actividad módulo 10 - mlops en la nube
# entrenamiento y serialización del modelo (breast cancer wisconsin)


import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# base_dir apunta a la carpeta donde está este archivo, no a donde ejecutes el script
base_dir = Path(__file__).parent.resolve()
ruta_salida = base_dir / "resultados"
ruta_salida.mkdir(parents=True, exist_ok=True)

def entrenar_y_guardar():
    # cargo dataset de sklearn (no requiere descarga externa)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    nombres = list(data.feature_names)

    # separo en train y test para validar rápido
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline sencillo: estandarización + regresión logística
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, n_jobs=None))
    ])

    # entreno
    pipe.fit(X_train, y_train)

    # evaluación simple para dejar registro en consola
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"accuracy: {acc:.4f}")
    print(f"roc_auc: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    # guardo el modelo y metadatos en carpeta resultados
    modelo_path = ruta_salida / "modelo.joblib"
    meta_path = ruta_salida / "metadata.json"

    joblib.dump(pipe, modelo_path)

    metadata = {
        "feature_names": nombres,
        "target_names": list(data.target_names),
        "task": "clasificacion binaria",
        "modelo": "pipeline(StandardScaler + LogisticRegression)",
        "metricas": {"accuracy_test": float(acc), "roc_auc_test": float(auc)}
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"modelo guardado en: {modelo_path.resolve()}")
    print(f"metadata guardada en: {meta_path.resolve()}")

if __name__ == "__main__":
    entrenar_y_guardar()