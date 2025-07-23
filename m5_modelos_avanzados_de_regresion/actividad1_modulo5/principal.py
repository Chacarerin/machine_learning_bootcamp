# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
import warnings
warnings.filterwarnings("ignore")

# CARGA Y PREPROCESAMIENTO DEL DATASET

# Creamos una función que descarga el dataset Adult desde OpenML y lo devuelve como un dataframe
def cargar_datos():
    print("Cargando datos desde OpenML...")
    datos = fetch_openml(name='adult', version=2, as_frame=True)
    df = datos.frame
    print(f"Se cargaron {df.shape[0]} filas y {df.shape[1]} columnas")
    return df

# Creamos una función que limpia los datos, transforma la variable objetivo en binaria,
# y aplica escalamiento y codificación a las variables predictoras
def preprocesar_datos(df):
    df = df.replace("?", np.nan).dropna()
    df['target'] = (df['class'] == '>50K').astype(int)
    X = df.drop(['class', 'target'], axis=1)
    y = df['target']
    
    columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocesador = ColumnTransformer(transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
    ])
    
    X_preprocesado = preprocesador.fit_transform(X)
    return X_preprocesado, y

# ENTRENAMIENTO DE MODELOS

# Creamos una función que entrena todos los modelos indicados en la actividad
# y devuelve un diccionario con los modelos ya entrenados
def entrenar_modelos(X_train, y_train):
    modelos = {}
    
    modelos['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
    modelos['Quantile10'] = QuantileRegressor(quantile=0.1, alpha=0)
    modelos['Quantile50'] = QuantileRegressor(quantile=0.5, alpha=0)
    modelos['Quantile90'] = QuantileRegressor(quantile=0.9, alpha=0)
    modelos['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    modelos['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    modelos_entrenados = {}
    
    # recorremos todos los modelos definidos y los entrenamos con los datos de entrenamiento
    for nombre, modelo in modelos.items():
        print(f"Entrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)
        modelos_entrenados[nombre] = modelo
        
    return modelos_entrenados

# EVALUACIÓN DE DESEMPEÑO

# Creamos una función que evalúa cada modelo con las métricas correspondientes:
# usa RMSE para regresión y accuracy + matriz de confusión para clasificación
def evaluar_modelos(modelos, X_test, y_test):
    print("\nEvaluación de los modelos")
    
    # recorremos cada modelo entrenado para aplicar la evaluación
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        
        # si el modelo es de regresión, usamos RMSE como métrica
        if 'Quantile' in nombre or 'Elastic' in nombre:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"{nombre} - RMSE: {rmse:.4f}")
        # si el modelo es clasificador, usamos accuracy y mostramos la matriz de confusión
        else:
            acc = accuracy_score(y_test, y_pred.round())
            print(f"{nombre} - Accuracy: {acc:.4f}")
            cm = confusion_matrix(y_test, y_pred.round())
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f"Matriz de Confusión - {nombre}")
            plt.show()

# FLUJO PRINCIPAL

# Creamos el flujo principal que ejecuta todo el proceso: carga, limpieza, entrenamiento y evaluación
if __name__ == "__main__":
    inicio = time.time()

    df = cargar_datos()
    X, y = preprocesar_datos(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelos = entrenar_modelos(X_train, y_train)
    evaluar_modelos(modelos, X_test, y_test)

# CURVA ROC E IMPORTANCIA DE VARIABLES
# Avanzamos con los puntos 4 y 5 de la actividad, que consisten en entrenar los modelos y evaluar su desempeño.
# Creamos una función que grafica la curva ROC para clasificadores binarios
def mostrar_curva_roc(modelos, X_test, y_test):
    from sklearn.metrics import roc_curve, auc

    for nombre in ['RandomForest', 'XGBoost']:
        if nombre in modelos:
            modelo = modelos[nombre]
            if hasattr(modelo, "predict_proba"):
                y_scores = modelo.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Creamos una función que muestra la importancia de las variables para modelos tipo árbol
def mostrar_importancia_variables(modelos, X_train):
    importances = {}
    nombres = ['RandomForest', 'XGBoost']

    for nombre in nombres:
        if nombre in modelos:
            modelo = modelos[nombre]
            if hasattr(modelo, "feature_importances_"):
                importances[nombre] = modelo.feature_importances_

    # los nombres de las variables ya codificadas son muchos, así que usamos índices
    for nombre in importances:
        valores = importances[nombre]
        indices = np.argsort(valores)[-10:][::-1]
        plt.bar(range(len(indices)), valores[indices])
        plt.title(f'Importancia de variables - {nombre}')
        plt.xlabel('Índice variable (codificada)')
        plt.ylabel('Importancia')
        plt.xticks(range(len(indices)), indices)
        plt.show()

# mostramos curvas ROC para los clasificadores
mostrar_curva_roc(modelos, X_test, y_test)

# mostramos importancia de variables para Random Forest y XGBoost
mostrar_importancia_variables(modelos, X_train) 

# Medimos el tiempo total de ejecución solo para custionarnos si debemos cambiar de equipo :)
fin = time.time()
duracion = fin - inicio
print(f"\nTiempo total de ejecución: {duracion:.2f} segundos")