from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import time

# Ajuste con Grid Search (explora combinaciones definidas)
def grid_search_rf(x_train, y_train):
    parametros = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    modelo = RandomForestClassifier(random_state=42)

    inicio = time.time()
    grid = GridSearchCV(modelo, parametros, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    fin = time.time()

    mejor_modelo = grid.best_estimator_
    duracion = fin - inicio

    return mejor_modelo, grid.best_params_, duracion

# Ajuste con Random Search (prueba combinaciones al azar)
def random_search_rf(x_train, y_train):
    distribuciones = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(3, 15)),
        'min_samples_split': randint(2, 11)
    }

    modelo = RandomForestClassifier(random_state=42)

    inicio = time.time()
    random_search = RandomizedSearchCV(
        modelo,
        distribuciones,
        n_iter=10,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(x_train, y_train)
    fin = time.time()

    mejor_modelo = random_search.best_estimator_
    duracion = fin - inicio

    return mejor_modelo, random_search.best_params_, duracion