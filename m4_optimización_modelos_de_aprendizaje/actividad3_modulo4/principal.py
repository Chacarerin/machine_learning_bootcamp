# carga de librerias necesarias

import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

from skopt import BayesSearchCV
from skopt.space import Integer

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Cargar el dataset de cáncer de mama
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Entrenar modelo base sin ajustar hiperparámetros
print("\nEntrenando modelo base...")
modelo_base = RandomForestClassifier(random_state=42)
modelo_base.fit(X_train, y_train)
y_pred_base = modelo_base.predict(X_test)
f1_base = f1_score(y_test, y_pred_base)

print(classification_report(y_test, y_pred_base))
print(f"F1-Score modelo base: {f1_base:.4f}")

# Optimización con Scikit-Optimize
print("\nOptimizando con Scikit-Optimize...")
inicio_skopt = time.time()

espacio_skopt = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(2, 20),
    'min_samples_split': Integer(2, 10),
}

optimizador_skopt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=espacio_skopt,
    n_iter=25,
    scoring='f1',
    cv=3,
    random_state=42,
    n_jobs=-1
)

optimizador_skopt.fit(X_train, y_train)
mejor_skopt = optimizador_skopt.best_estimator_
y_pred_skopt = mejor_skopt.predict(X_test)
f1_skopt = f1_score(y_test, y_pred_skopt)
fin_skopt = time.time()

print("Mejores parámetros (Skopt):", optimizador_skopt.best_params_)
print(f"F1-Score (Skopt): {f1_skopt:.4f}")
print(f"Tiempo Skopt: {fin_skopt - inicio_skopt:.2f} segundos")

# Optimización con Hyperopt
print("\nOptimizando con Hyperopt...")

def objetivo(params):
    params = {k: int(v) for k, v in params.items()}
    modelo = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42
    )
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    f1 = f1_score(y_test, pred)
    return {'loss': -f1, 'status': STATUS_OK}

espacio_hyperopt = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
    'max_depth': hp.quniform('max_depth', 2, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
}

trials = Trials()
inicio_hyperopt = time.time()
mejores_params = fmin(
    fn=objetivo,
    space=espacio_hyperopt,
    algo=tpe.suggest,
    max_evals=25,
    trials=trials,
    rstate=np.random.default_rng(42)
)
fin_hyperopt = time.time()

mejores_params = {k: int(v) for k, v in mejores_params.items()}
print("Mejores parámetros (Hyperopt):", mejores_params)

modelo_hyperopt = RandomForestClassifier(**mejores_params, random_state=42)
modelo_hyperopt.fit(X_train, y_train)
y_pred_hyperopt = modelo_hyperopt.predict(X_test)
f1_hyperopt = f1_score(y_test, y_pred_hyperopt)

print(f"F1-Score (Hyperopt): {f1_hyperopt:.4f}")
print(f"Tiempo Hyperopt: {fin_hyperopt - inicio_hyperopt:.2f} segundos")

# Comparar resultados
print("\nResumen comparativo:")
print(f"Modelo base       F1-Score: {f1_base:.4f}")
print(f"Scikit-Optimize   F1-Score: {f1_skopt:.4f}")
print(f"Hyperopt          F1-Score: {f1_hyperopt:.4f}")

print(f"\nTiempo Skopt   : {fin_skopt - inicio_skopt:.2f} s")
print(f"Tiempo Hyperopt: {fin_hyperopt - inicio_hyperopt:.2f} s")