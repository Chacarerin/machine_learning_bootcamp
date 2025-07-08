import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# -----------------------------
# Dataset y preprocesamiento
# -----------------------------
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Modelo base sin optimización
# -----------------------------
def modelo_base():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print("\n--- Modelo base ---")
    print("F1 Score:", f1)

# -----------------------------
# Optimización con Optuna
# -----------------------------
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

def optimizar_optuna():
    print("\n--- Optuna ---")
    start = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor F1 Score:", study.best_value)
    print(f"Tiempo de ejecución: {time.time() - start:.2f} segundos")

# -----------------------------
# Optimización con Ray Tune
# -----------------------------
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    tune.report(f1=f1)  # ✅ Usa el nombre 'f1'

def optimizar_ray():
    print("\n--- Ray Tune ---")
    search_space = {
        "n_estimators": tune.randint(50, 300),
        "max_depth": tune.randint(2, 20),
        "min_samples_split": tune.randint(2, 20),
    }
    scheduler = ASHAScheduler(metric="f1", mode="max")  # ✅ Ajustado
    start = time.time()
    analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=20,
        scheduler=scheduler,
        verbose=1
    )
    print("Mejores hiperparámetros:", analysis.best_config)
    print("Mejor F1 Score:", analysis.best_result["f1"])
    print(f"Tiempo de ejecución: {time.time() - start:.2f} segundos")

# -----------------------------
# Ejecución principal
# -----------------------------
if __name__ == "__main__":
    modelo_base()
    optimizar_optuna()
    optimizar_ray()