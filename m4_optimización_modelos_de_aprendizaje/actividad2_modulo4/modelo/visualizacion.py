import matplotlib.pyplot as plt
import numpy as np

# Muestra la importancia de las variables del modelo
def graficar_importancias(modelo, nombres_columnas):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importancias)), importancias[indices], align='center')
    plt.yticks(range(len(importancias)), [nombres_columnas[i] for i in indices])
    plt.title("Importancia de características")
    plt.tight_layout()
    plt.savefig("grafico_comparativo2.jpeg")
    plt.close()

# Compara F1 y accuracy entre modelos
def graficar_f1_accuracy(accs, f1s):
    modelos = ['Base', 'Grid', 'Random']

    x = np.arange(len(modelos))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, accs, width, label='Accuracy')
    plt.bar(x + width/2, f1s, width, label='F1 Score')

    plt.ylabel("Puntaje")
    plt.title("Comparación de modelos")
    plt.xticks(x, modelos)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_comparativo1.jpeg")
    plt.close()