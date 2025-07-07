from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

def grid_search_rf(x_train, y_train):
    parametros = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), parametros, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.best_estimator_, grid.best_params_

def random_search_rf(x_train, y_train):
    distribuciones = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(3, 15)),
        'min_samples_split': randint(2, 11)
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        distribuciones,
        n_iter=20,
        scoring='f1',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(x_train, y_train)
    return random_search.best_estimator_, random_search.best_params_