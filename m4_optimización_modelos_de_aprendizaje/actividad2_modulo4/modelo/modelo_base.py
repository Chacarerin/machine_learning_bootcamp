from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# Entrena modelo Random Forest sin ajustes
def entrenar_modelo_base(x_train, y_train):
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(x_train, y_train)
    return modelo

# Evalúa el modelo con varias métricas
def evaluar_modelo(modelo, x_test, y_test):
    pred = modelo.predict(x_test)
    prob = modelo.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    recall = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    # Mostrar resultados
    print("\n[Evaluación del Modelo]")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, f1