# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import time
import warnings
warnings.filterwarnings("ignore")

# CARGA Y PREPROCESAMIENTO DEL DATASET

# Creamos una función que carga y prepara el dataset 'credit-g' de OpenML
def cargar_datos():
    df = fetch_openml(name='credit-g', version=1, as_frame=True).frame
    df = df.dropna()
    df['target'] = (df['class'] == 'good').astype(int)
    X = df.drop(['class', 'target'], axis=1)
    y = df['target']
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), X.columns

# ENTRENAMIENTO Y EVALUACIÓN DEL MODELO

# Creamos una función que entrena un modelo Lasso (Regresión Logística L1),
# evalúa su rendimiento e imprime las métricas
def entrenar_evaluar_modelo(X_train, X_test, y_train, y_test):
    print("\nEntrenando modelo Lasso (Regresión Logística L1)...")
    modelo = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Matriz de Confusión")
    plt.savefig("matriz_confusion.png")
    plt.close()

    return modelo

# INTERPRETABILIDAD CON SHAP

# Creamos una función que utiliza SHAP para interpretar los resultados del modelo
# Solo se incluye el summary_plot ya que force_plot no se puede guardar fuera de Jupyter
def interpretar_con_shap(modelo, X_train, feature_names):
    print("\nGenerando explicaciones SHAP...")
    explainer = shap.LinearExplainer(modelo, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_train)

    # Plot resumen global de importancia de características
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()

# FLUJO PRINCIPAL

# Este bloque ejecuta todo el proceso
if __name__ == "__main__":
    inicio = time.time()

    (X_train, X_test, y_train, y_test), feature_names = cargar_datos()
    modelo = entrenar_evaluar_modelo(X_train, X_test, y_train, y_test)
    interpretar_con_shap(modelo, X_train, feature_names)

    fin = time.time()
    print(f"\nTiempo total de ejecución: {fin - inicio:.2f} segundos")
