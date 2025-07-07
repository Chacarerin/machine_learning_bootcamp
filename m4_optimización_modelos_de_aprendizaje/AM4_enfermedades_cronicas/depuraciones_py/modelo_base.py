# preparamos ahora el script para el modelo base, que entrenará un modelo de Random Forest y evaluará su rendimiento.
# Este script cargará los datos preprocesados, entrenará el modelo y generará métricas de 
# evaluación como accuracy, matriz de confusión y ROC AUC.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv').values.ravel()
y_test = pd.read_csv('./data/y_test.csv').values.ravel()

# Entrenar modelo base
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)

# Evaluación
print("=== Evaluación del Modelo Base ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, annot=False, cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('./data/matriz_confusion_base.png')  # Guarda la imagen
plt.close()

# Calcular ROC-AUC (binarizando clases múltiples)
y_test_bin = label_binarize(y_test, classes=list(range(len(set(y_test)))))
y_prob_bin = y_prob

if y_test_bin.shape[1] > 1:
    auc = roc_auc_score(y_test_bin, y_prob_bin, average='macro', multi_class='ovr')
    print("ROC AUC Score:", auc)
else:
    print("ROC AUC Score no aplicable con una sola clase.")