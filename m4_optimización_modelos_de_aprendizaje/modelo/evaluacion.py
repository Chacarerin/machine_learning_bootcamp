from .modelo_base import evaluar_modelo

def comparar_modelos(modelo_base, modelo_grid, modelo_random, x_test, y_test):
    print("\n=== Modelo Base ===")
    evaluar_modelo(modelo_base, x_test, y_test)

    print("\n=== Modelo con Grid Search ===")
    evaluar_modelo(modelo_grid, x_test, y_test)

    print("\n=== Modelo con Random Search ===")
    evaluar_modelo(modelo_random, x_test, y_test)