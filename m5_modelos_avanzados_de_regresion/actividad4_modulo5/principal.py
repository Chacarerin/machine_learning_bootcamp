# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve
)
import time
import warnings
warnings.filterwarnings("ignore")

# CARGA Y PREPROCESAMIENTO DEL DATASET

# Creamos una función que carga el dataset Adult Income desde OpenML,
# elimina valores faltantes, convierte la variable objetivo en binaria
# y aplica codificación one-hot para las variables categóricas
def cargar_datos():
    df = fetch_openml(name='adult', version=2, as_frame=True).frame
    df = df.replace("?", np.nan).dropna()
    df['target'] = (df['class'] == '>50K').astype(int)
    X = df.drop(['class', 'target'], axis=1)
    y = df['target']
    X = pd.get_dummies(X)
    return X, y

# VALIDACIÓN CRUZADA Y EVALUACIÓN

# Creamos una función que aplica validación cruzada con la estrategia indicada,
# entrena un modelo de regresión logística, y calcula métricas y visualizaciones:
# matriz de confusión, curva ROC y curva Precision-Recall
def evaluar_modelo(nombre, modelo, X, y, cv_strategy):
    print(f"\nEvaluando con {nombre}...")
    y_pred = cross_val_predict(modelo, X, y, cv=cv_strategy, method="predict")
    y_prob = cross_val_predict(modelo, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"{nombre} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.savefig(f"Figure_Matriz_{nombre}.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Curva ROC - {nombre}')
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdaderos Positivos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figure_ROC_{nombre}.png")
    plt.close()

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'{nombre}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {nombre}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figure_PR_{nombre}.png")
    plt.close()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }

# FLUJO PRINCIPAL

# Este bloque ejecuta todo el flujo: carga de datos, validación cruzada con distintas estrategias,
# evaluación del modelo, generación de métricas y visualización de resultados.
if __name__ == "__main__":
    inicio = time.time()

    X, y = cargar_datos()
    modelo = LogisticRegression(max_iter=1000)

    resultados = {}

    resultados['KFold'] = evaluar_modelo(
        "KFold", modelo, X, y, KFold(n_splits=5, shuffle=True, random_state=42)
    )

    resultados['StratifiedKFold'] = evaluar_modelo(
        "StratifiedKFold", modelo, X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )

    resultados['LOO'] = evaluar_modelo(
        "LeaveOneOut", modelo, X.sample(n=500, random_state=42), y.sample(n=500, random_state=42),
        LeaveOneOut()
    )

# aquí vemos si necesitamos cambiar o no de computador
    fin = time.time()
    duracion = fin - inicio
    print(f"\nTiempo total de ejecución: {duracion:.2f} segundos")