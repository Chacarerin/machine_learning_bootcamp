from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def entrenar_modelo_base(x_train, y_train):
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(x_train, y_train)
    return modelo

def evaluar_modelo(modelo, x_test, y_test):
    pred = modelo.predict(x_test)
    prob = modelo.predict_proba(x_test)[:, 1]

    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print("\n[Evaluación del Modelo]")
    print(f"F1: {f1:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")