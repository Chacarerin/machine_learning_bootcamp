# train_model.py
# reentrena el modelo (wine) y sobrescribe el artefacto en resultados_sesion2/modelo.pkl

from pathlib import Path
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
N_ESTIMATORS = 300
CARPETA_RESULTADOS = "resultados_sesion2"
NOMBRE_MODELO = "modelo.pkl"

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

    paquete = {
        "model": clf,
        "class_names": list(data.target_names),
        "feature_names": list(data.feature_names),
    }
    joblib.dump(paquete, modelo_path)

    with open(resultados / "metricas_reentrenamiento.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy_test: {acc:.4f}\n")

    print("modelo reentrenado. guardado en:", modelo_path)

if __name__ == "__main__":
    main()
