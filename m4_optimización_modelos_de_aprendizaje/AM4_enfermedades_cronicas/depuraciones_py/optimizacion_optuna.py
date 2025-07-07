# Utilizamos Optuna para la optimización de hiperparámetros
# Este script optimiza un modelo de Random Forest utilizando Optuna para encontrar los mejores hiperparámetros.

import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Cargar datos
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv').values.ravel()

# Función objetivo para Optuna
def objective(trial):
    # Hiperparámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Evaluar con validación cruzada
    scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    return scores.mean()

# Crear estudio
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Mostrar mejores resultados
print("Mejores hiperparámetros encontrados:")
print(study.best_params)
print(f"Mejor accuracy promedio: {study.best_value:.4f}")

# Entrenar modelo final con mejores parámetros
best_params = study.best_params
modelo_opt = RandomForestClassifier(**best_params, random_state=42)
modelo_opt.fit(X_train, y_train)

# Evaluar con X_test
X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv').values.ravel()
y_pred = modelo_opt.predict(X_test)

print("\nEvaluación del modelo optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred))