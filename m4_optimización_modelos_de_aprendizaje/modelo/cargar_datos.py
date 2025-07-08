import pandas as pd

def cargar_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columnas = ['embarazos', 'glucosa', 'presion', 'pliegue', 'insulina', 'imc', 'dpf', 'edad', 'resultado']
    df = pd.read_csv(url, header=None, names=columnas)
    return df

def explorar_dataset(df):
    print("Dimensiones:", df.shape)
    print("\nPrimeras filas:\n", df.head())
    print("\nValores cero en columnas críticas:")
    for col in ['glucosa', 'presion', 'pliegue', 'insulina', 'imc']:
        print(f"{col}: {(df[col] == 0).sum()} ceros")