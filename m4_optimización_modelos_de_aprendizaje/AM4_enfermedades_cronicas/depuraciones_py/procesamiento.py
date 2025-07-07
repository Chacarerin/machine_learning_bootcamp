# Ahora vamos a preparar los datos para el modelo de Machine Learning.
# Esto incluye la separación de variables independientes y dependientes, la codificación de la variable objetivo
# y la división del dataset en conjuntos de entrenamiento y prueba.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar datos
df = pd.read_csv('./data/Training.csv')

# Eliminar columna vacía
if 'Unnamed: 133' in df.columns:
    df.drop(columns=['Unnamed: 133'], inplace=True)

# Separar variables independientes (X) y dependiente (y)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# División en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Verificar shapes
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_test: ", y_test.shape)

# Guardar los sets para uso posterior
X_train.to_csv('./data/X_train.csv', index=False)
X_test.to_csv('./data/X_test.csv', index=False)
pd.DataFrame(y_train, columns=['prognosis']).to_csv('./data/y_train.csv', index=False)
pd.DataFrame(y_test, columns=['prognosis']).to_csv('./data/y_test.csv', index=False)