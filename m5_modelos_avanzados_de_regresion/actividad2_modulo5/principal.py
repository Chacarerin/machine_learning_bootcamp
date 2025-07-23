# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from statsmodels.tsa.api import VAR
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

# 1. CARGA Y PREPARACIÓN DE LOS DATOS

def cargar_datos_california():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return X, y

def cargar_datos_adult():
    data = fetch_openml(name='adult', version=2, as_frame=True)
    df = data.frame
    df = df.replace("?", np.nan).dropna()
    df['target'] = (df['class'] == '>50K').astype(int)
    X = df.drop(['class', 'target'], axis=1)
    y = df['target']
    X = pd.get_dummies(X)
    return X, y

def cargar_datos_macro():
    from statsmodels.datasets.macrodata import load_pandas
    data = load_pandas().data
    df = data[["realgdp", "realcons", "realinv"]]
    df.index = pd.date_range(start="1959Q1", periods=len(df), freq="Q")
    return df

# 2. ENTRENAMIENTO DE MODELOS

def entrenar_elastic_net(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    modelo = ElasticNet(alpha=0.1, l1_ratio=0.5)
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Elastic Net - RMSE: {rmse:.4f}")
    print("\nCoeficientes del modelo Elastic Net:")
    for nombre, valor in zip(X.columns, modelo.coef_):
        print(f"{nombre}: {valor:.4f}")
    return modelo

def entrenar_regresion_cuantilica(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    resultados = {}
    predicciones = pd.DataFrame(index=y_test.index)
    for q in [0.1, 0.5, 0.9]:
        modelo = QuantileRegressor(quantile=q, alpha=0)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        loss = mean_pinball_loss(y_test, y_pred, alpha=q)
        print(f"Quantile {int(q*100)} - Pinball Loss: {loss:.4f}")
        predicciones[f"q{int(q*100)}"] = y_pred

    # Graficar comparativa
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=predicciones.head(100))
    plt.title("Predicciones de Regresión Cuantílica (primeros 100 registros)")
    plt.xlabel("Índice")
    plt.ylabel("Probabilidad estimada")
    plt.legend(title="Percentil")
    plt.tight_layout()
    plt.savefig("Figure_1.png")
    plt.show()
    return resultados

def entrenar_var(df):
    modelo = VAR(df)
    resultados = modelo.select_order(10)
    lag_optimo = resultados.selected_orders['aic']
    var_model = modelo.fit(lag_optimo)
    pred = var_model.forecast(df.values[-lag_optimo:], steps=5)
    pred_df = pd.DataFrame(pred, columns=df.columns)
    print("\nProyección VAR (5 pasos):")
    print(pred_df)

    # Graficar predicciones
    pred_df.plot(figsize=(10, 5), title="Proyección VAR - 5 pasos adelante")
    plt.tight_layout()
    plt.savefig("Figure_2.png")
    plt.show()
    return var_model, pred_df

# 3. FLUJO PRINCIPAL

if __name__ == "__main__":
    inicio = time.time()

    X_cal, y_cal = cargar_datos_california()
    entrenar_elastic_net(X_cal, y_cal)

    X_adult, y_adult = cargar_datos_adult()
    entrenar_regresion_cuantilica(X_adult, y_adult)

    df_macro = cargar_datos_macro()
    entrenar_var(df_macro)

# solo por curiosidad, medimos el tiempo total de ejecución
    fin = time.time()
    duracion = fin - inicio
    print(f"\nTiempo total de ejecución: {duracion:.2f} segundos")
