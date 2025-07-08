import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import optuna
import time

# 1. CARGA Y EXPLORACIÓN INICIAL

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columnas = ['embarazos', 'glucosa', 'presion', 'pliegue_cutaneo', 'insulina', 'imc', 'pedigree', 'edad', 'diabetes']
df = pd.read_csv(url, header=None, names=columnas)

print("Dimensiones del dataset:", df.shape)
print("Primeras filas:")
print(df.head())

# Detección de ceros que pueden representar valores faltantes
columnas_con_ceros = ['glucosa', 'presion', 'pliegue_cutaneo', 'insulina', 'imc']
print("\nValores cero por columna:")
print((df[columnas_con_ceros] == 0).sum())

# Reemplazo de ceros por la mediana (salvo embarazos y diabetes)
for col in columnas_con_ceros:
    df[col] = df[col].replace(0, df[col].median())

# 2. PREPROCESAMIENTO

X = df.drop('diabetes', axis=1)
y = df['diabetes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 3. MODELO BASE: RANDOM FOREST

modelo_base = RandomForestClassifier(random_state=42)
modelo_base.fit(X_train, y_train)
y_pred_base = modelo_base.predict(X_test)

def evaluar_modelo(y_test, y_pred, nombre="Modelo"):
    print(f"\nEvaluación del {nombre}")
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Precisión:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_pred))
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

evaluar_modelo(y_test, y_pred_base, "Modelo Base")

# 4. AJUSTE DE HIPERPARÁMETROS

# GRID SEARCH
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3, scoring='f1', n_jobs=-1)

inicio_grid = time.time()
grid_search.fit(X_train, y_train)
tiempo_grid = time.time() - inicio_grid

print("\nMejores hiperparámetros - Grid Search:")
print(grid_search.best_params_)

y_pred_grid = grid_search.predict(X_test)
evaluar_modelo(y_test, y_pred_grid, "Grid Search")

# OPTUNA: Búsqueda Bayesiana

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return f1_score(y_test, y_pred)

inicio_optuna = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
tiempo_optuna = time.time() - inicio_optuna

print("\nMejores hiperparámetros - Optuna:")
print(study.best_params)

modelo_optuna = RandomForestClassifier(**study.best_params, random_state=42)
modelo_optuna.fit(X_train, y_train)
y_pred_optuna = modelo_optuna.predict(X_test)
evaluar_modelo(y_test, y_pred_optuna, "Optuna")

# 5. VISUALIZACIÓN

scores = {
    "Modelo": ["Base", "GridSearch", "Optuna"],
    "F1-Score": [
        f1_score(y_test, y_pred_base),
        f1_score(y_test, y_pred_grid),
        f1_score(y_test, y_pred_optuna)
    ],
    "Tiempo (s)": [
        0,  # modelo base
        round(tiempo_grid, 2),
        round(tiempo_optuna, 2)
    ]
}

df_scores = pd.DataFrame(scores)

plt.figure(figsize=(8, 5))
sns.barplot(x="Modelo", y="F1-Score", data=df_scores)
plt.title("Comparación de F1-Score por Técnica")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Modelo", y="Tiempo (s)", data=df_scores)
plt.title("Comparación de Tiempos de Ejecución")
plt.grid(True)
plt.tight_layout()
plt.show()

