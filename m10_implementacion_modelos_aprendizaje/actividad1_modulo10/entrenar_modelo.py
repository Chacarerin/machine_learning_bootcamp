# entrenar_modelo.py
# reentrena el modelo iris y sobrescribe el archivo de artefacto en resultados_sesion1/modelo.pkl

from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
N_ESTIMATORS = 200
CARPETA_RESULTADOS = "resultados_sesion1"
NOMBRE_MODELO = "modelo.pkl"

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

    joblib.dump({"model": clf, "target_names": iris.target_names.tolist()}, resultados / NOMBRE_MODELO)

    acc = accuracy_score(yte, clf.predict(xte))
    with open(resultados / "metricas_reentrenamiento.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy_test: {acc:.4f}\n")

    print("modelo reentrenado y guardado en:", resultados / NOMBRE_MODELO)

if __name__ == "__main__":
    main()
