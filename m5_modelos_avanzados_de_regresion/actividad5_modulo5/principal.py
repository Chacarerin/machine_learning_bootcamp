# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings("ignore")

# CARGA Y PREPROCESAMIENTO DEL DATASET

# Creamos una función que carga el dataset Adult Income,
# elimina valores nulos, codifica variables categóricas y escala los datos
def cargar_datos():
    df = fetch_openml(name='adult', version=2, as_frame=True).frame
    df = df.replace("?", np.nan).dropna()
    df['target'] = (df['class'] == '>50K').astype(int)
    X = df.drop(['class', 'target'], axis=1)
    y = df['target']
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), X.columns

# ENTRENAMIENTO Y EVALUACIÓN DE LOS MODELOS

# Creamos una función que entrena un modelo, evalúa su rendimiento,
# imprime el RMSE y retorna los coeficientes
def entrenar_modelo(nombre, modelo, X_train, X_test, y_train, y_test, feature_names):
    print(f"\nEntrenando modelo: {nombre}")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{nombre} - RMSE: {rmse:.4f}")
    coef = modelo.coef_
    return pd.Series(coef, index=feature_names, name=nombre), rmse

# Creamos una función que grafica los coeficientes de los modelos
def graficar_coeficientes(resultados):
    df_coef = pd.concat(resultados, axis=1)
    df_coef.plot(kind="bar", figsize=(12, 6))
    plt.title("Comparación de Coeficientes por Modelo")
    plt.xlabel("Variables")
    plt.ylabel("Valor del Coeficiente")
    plt.tight_layout()
    plt.savefig("Figure_Coeficientes.png")
    plt.close()

# FLUJO PRINCIPAL

# Este bloque ejecuta todo: carga de datos, entrenamiento, evaluación y visualización
if __name__ == "__main__":
    inicio = time.time()

    (X_train, X_test, y_train, y_test), feature_names = cargar_datos()

    resultados = {}
    errores = {}

    # Entrenamos y evaluamos Lasso
    coef_lasso, rmse_lasso = entrenar_modelo(
        "Lasso", Lasso(alpha=0.1), X_train, X_test, y_train, y_test, feature_names)
    resultados["Lasso"] = coef_lasso
    errores["Lasso"] = rmse_lasso

    # Entrenamos y evaluamos Ridge
    coef_ridge, rmse_ridge = entrenar_modelo(
        "Ridge", Ridge(alpha=1.0), X_train, X_test, y_train, y_test, feature_names)
    resultados["Ridge"] = coef_ridge
    errores["Ridge"] = rmse_ridge

    # Entrenamos y evaluamos ElasticNet
    coef_elastic, rmse_elastic = entrenar_modelo(
        "ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5), X_train, X_test, y_train, y_test, feature_names)
    resultados["ElasticNet"] = coef_elastic
    errores["ElasticNet"] = rmse_elastic

    # Visualizamos los coeficientes
    graficar_coeficientes(resultados)

    # Mostramos resumen de errores
    print("\nResumen de errores RMSE:")
    for modelo, error in errores.items():
        print(f"{modelo}: {error:.4f}")

    # Tiempo total de ejecución
    fin = time.time()
    duracion = fin - inicio
    print(f"\nTiempo total de ejecución: {duracion:.2f} segundos")
