#primero preparamos los datos utilizando modulos de python como pandas:

import pandas as pd

# Cargar datos
df = pd.read_csv('./data/Training.csv')

# Eliminar columna vacía
df.drop(columns=['Unnamed: 133'], inplace=True)

# Mostrar distribución de la variable objetivo
print("\nDistribución de enfermedades (prognosis):")
print(df['prognosis'].value_counts())

# Mostrar cuántas clases hay
# Nota: error inicialmente por imprimir df['Disease'] pero la columna correcta es df['prognosis'].

print(f"\nNúmero de enfermedades distintas: {df['prognosis'].nunique()}") 