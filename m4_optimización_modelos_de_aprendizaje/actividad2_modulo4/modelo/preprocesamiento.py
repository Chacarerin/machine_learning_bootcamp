import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Limpia los ceros, escala y divide en train/test (70/30)
def limpiar_y_dividir(df):
    columnas_con_ceros = ['glucosa', 'presion', 'pliegue', 'insulina', 'imc']
    
    # Reemplazo de ceros por la mediana
    for col in columnas_con_ceros:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop('resultado', axis=1)
    y = df['resultado']

    # Escalado de variables
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    # División 70% entrenamiento, 30% prueba
    x_train, x_test, y_train, y_test = train_test_split(
        X_escalado, y, test_size=0.3, random_state=42, stratify=y
    )

    return x_train, x_test, y_train, y_test