# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
import warnings
warnings.filterwarnings("ignore")

# 1. CARGA Y PREPROCESAMIENTO DE LOS DATOS

# Creamos una función que carga el dataset Adult desde OpenML y lo preprocesa
def cargar_datos():
    data = fetch_openml(name='adult', version=2, as_frame=True)
    df = data.frame
    df = df.replace("?", np.nan).dropna()
    df['target'] = (df['class'] == '>50K').astype(int)
    X = df.drop(['class', 'target'], axis=1)
    y = df['target']
    X = pd.get_dummies(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 2. ENTRENAMIENTO DE MODELOS

# Creamos una función que entrena los modelos indicados y devuelve resultados
def entrenar_modelos(X_train, X_test, y_train, y_test):
    modelos = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        print(f"Entrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': acc,
            'matriz': cm
        }
        print(f"{nombre} - Accuracy: {acc:.4f}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Matriz de Confusión - {nombre}")
        plt.savefig(f"Figure_{nombre}.png")
        plt.show()

    return resultados

# Creamos una función que genera una visualización comparativa de accuracy
def comparar_accuracy(resultados):
    nombres = []
    valores = []
    for nombre, datos in resultados.items():
        nombres.append(nombre)
        valores.append(datos['accuracy'])

    plt.figure(figsize=(8, 5))
    sns.barplot(x=nombres, y=valores)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Comparación de Accuracy entre Modelos')
    plt.tight_layout()
    plt.savefig("Figure_Accuracy.png")
    plt.show()

# 3. FLUJO PRINCIPAL

# Este bloque ejecuta todo el proceso: carga, entrenamiento, evaluación y visualización
if __name__ == "__main__":
    inicio = time.time()

    X_train, X_test, y_train, y_test = cargar_datos()
    resultados = entrenar_modelos(X_train, X_test, y_train, y_test)
    comparar_accuracy(resultados)

    fin = time.time()
    duracion = fin - inicio
    print(f"\nTiempo total de ejecución: {duracion:.2f} segundos")
