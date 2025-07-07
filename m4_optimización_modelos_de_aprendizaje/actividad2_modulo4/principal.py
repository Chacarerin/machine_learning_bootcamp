from modelo.cargar_datos import cargar_dataset, explorar_dataset
from modelo.preprocesamiento import limpiar_y_dividir
from modelo.modelo_base import entrenar_modelo_base
from modelo.ajuste_hiperparametros import grid_search_rf, random_search_rf
from modelo.evaluacion import comparar_modelos
from modelo.visualizacion import graficar_importancias, graficar_f1_accuracy

from sklearn.metrics import f1_score

def main():
    # Cargar y explorar datos
    print("Cargando dataset...")
    df = cargar_dataset()
    explorar_dataset(df)

    # Preprocesar
    print("\nPreprocesando datos...")
    x_train, x_test, y_train, y_test = limpiar_y_dividir(df)

    # Entrenar modelo base
    print("\nEntrenando modelo base...")
    modelo_base = entrenar_modelo_base(x_train, y_train)

    # Ajuste con Grid Search
    print("\nEjecutando Grid Search...")
    modelo_grid, params_grid, duracion_grid = grid_search_rf(x_train, y_train)
    print("Parámetros óptimos (Grid):", params_grid)

    # Ajuste con Random Search
    print("\nEjecutando Random Search...")
    modelo_random, params_random, duracion_random = random_search_rf(x_train, y_train)
    print("Parámetros óptimos (Random):", params_random)

    # Comparar modelos
    accs, f1s = comparar_modelos(
        modelo_base, modelo_grid, modelo_random,
        x_test, y_test, duracion_grid, duracion_random
    )

    # Visualización
    nombres_columnas = ['embarazos', 'glucosa', 'presion', 'pliegue',
                        'insulina', 'imc', 'dpf', 'edad']
    
    graficar_importancias(modelo_grid, nombres_columnas)
    graficar_f1_accuracy(accs, f1s)

if __name__ == "__main__":
    main()