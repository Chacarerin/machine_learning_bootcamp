from .modelo_base import evaluar_modelo

# Compara los tres modelos y guarda tiempos de entrenamiento
def comparar_modelos(modelo_base, modelo_grid, modelo_random, x_test, y_test,
                     duracion_grid, duracion_random):
    
    print("\n=== Modelo Base ===")
    acc_b, f1_b = evaluar_modelo(modelo_base, x_test, y_test)

    print("\n=== Modelo con Grid Search ===")
    acc_g, f1_g = evaluar_modelo(modelo_grid, x_test, y_test)

    print("\n=== Modelo con Random Search ===")
    acc_r, f1_r = evaluar_modelo(modelo_random, x_test, y_test)

    # Guardar los tiempos en archivo de texto
    with open("tiempos_entrenamiento.txt", "w") as archivo:
        archivo.write(f"Duración Grid Search: {duracion_grid:.2f} segundos\n")
        archivo.write(f"Duración Random Search: {duracion_random:.2f} segundos\n")

    return (acc_b, acc_g, acc_r), (f1_b, f1_g, f1_r)