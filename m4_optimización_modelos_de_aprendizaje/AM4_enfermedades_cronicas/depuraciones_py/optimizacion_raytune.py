# Utilizamos ray tune para la optimización de hiperparámetros
# Este script optimiza un modelo de Random Forest utilizando Ray Tune para encontrar 
# los mejores hiperparámetros

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Obtener ruta absoluta al directorio de datos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Función de entrenamiento para Ray Tune
def train_model(config):
    try:
        # Cargar datos desde rutas absolutas
        X_train = pd.read_csv(os.path.join(BASE_DIR, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(BASE_DIR, 'y_train.csv')).values.ravel()
        X_test = pd.read_csv(os.path.join(BASE_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(BASE_DIR, 'y_test.csv')).values.ravel()

        # Crear y entrenar modelo
        model = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_split=int(config["min_samples_split"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Reportar resultado al scheduler
        tune.report({"accuracy": acc})

    except Exception as e:
        print("❌ Error en trial:", e)
        tune.report({"accuracy": 0})

# Espacio de búsqueda de hiperparámetros
search_space = {
    "n_estimators": tune.randint(50, 300),
    "max_depth": tune.randint(5, 50),
    "min_samples_split": tune.randint(2, 10),
    "min_samples_leaf": tune.randint(1, 5),
}

# Scheduler (con criterio definido aquí)
scheduler = ASHAScheduler(metric="accuracy", mode="max")

# Ejecución de Ray Tune
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=20,
    scheduler=scheduler,
    verbose=1
)

# Obtener mejores resultados
best_config = analysis.get_best_config(metric="accuracy", mode="max")
best_trial = analysis.get_best_trial(metric="accuracy", mode="max", scope="all")

# Mostrar resultados
print("==== Mejor resultado ====")
print("Mejores hiperparámetros encontrados por Ray Tune:")
print(best_config)
print("Mejor accuracy obtenido:", best_trial.metric_analysis["accuracy"]["max"])