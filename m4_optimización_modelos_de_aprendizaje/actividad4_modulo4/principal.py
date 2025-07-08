# principal.py

import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings("ignore")

# Cargar el dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en entrenamiento y prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Entrenar modelo base sin optimización
modelo_base = RandomForestClassifier(random_state=42)
modelo_base.fit(X_train, y_train)
y_pred_base = modelo_base.predict(X_test)
f1_base = f1_score(y_test, y_pred_base)

print("\nModelo base:")
print(classification_report(y_test, y_pred_base))
print(f"F1-Score modelo base: {f1_base:.4f}")

# Definir espacio de búsqueda: [n_estimators, max_depth, min_samples_split]
# Se codifican como enteros en rangos razonables
param_bounds = {
    "n_estimators": (10, 300),
    "max_depth": (2, 20),
    "min_samples_split": (2, 10)
}

# Preparar entorno genético con DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("n_estimators", random.randint, *param_bounds["n_estimators"])
toolbox.register("max_depth", random.randint, *param_bounds["max_depth"])
toolbox.register("min_samples_split", random.randint, *param_bounds["min_samples_split"])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.min_samples_split), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluación con cross_val_score (F1 como métrica)
def evaluar(individuo):
    params = {
        "n_estimators": individuo[0],
        "max_depth": individuo[1],
        "min_samples_split": individuo[2],
        "random_state": 42
    }
    modelo = RandomForestClassifier(**params)
    scores = cross_val_score(modelo, X_train, y_train, cv=3, scoring='f1')
    return np.mean(scores),

toolbox.register("evaluate", evaluar)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt,
                 low=[param_bounds["n_estimators"][0], param_bounds["max_depth"][0], param_bounds["min_samples_split"][0]],
                 up=[param_bounds["n_estimators"][1], param_bounds["max_depth"][1], param_bounds["min_samples_split"][1]],
                 indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Crear población inicial y ejecutar algoritmo genético
random.seed(42)
poblacion = toolbox.population(n=10)
NGEN = 15
HALL_OF_FAME = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

print("\nEjecutando algoritmo genético...")
poblacion, logbook = algorithms.eaSimple(
    poblacion, toolbox, cxpb=0.5, mutpb=0.2, ngen=NGEN,
    stats=stats, halloffame=HALL_OF_FAME, verbose=False
)

mejor = HALL_OF_FAME[0]
mejores_params = {
    "n_estimators": mejor[0],
    "max_depth": mejor[1],
    "min_samples_split": mejor[2],
    "random_state": 42
}

print(f"\nMejores hiperparámetros encontrados: {mejores_params}")

# Entrenar modelo final con mejores hiperparámetros
modelo_final = RandomForestClassifier(**mejores_params)
modelo_final.fit(X_train, y_train)
y_pred_final = modelo_final.predict(X_test)
f1_final = f1_score(y_test, y_pred_final)

print("\nModelo optimizado con algoritmo genético:")
print(classification_report(y_test, y_pred_final))
print(f"F1-Score modelo optimizado: {f1_final:.4f}")

# Comparación final
print("\nResumen comparativo:")
print(f"F1 modelo base      : {f1_base:.4f}")
print(f"F1 modelo optimizado: {f1_final:.4f}")