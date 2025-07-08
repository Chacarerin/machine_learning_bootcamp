import matplotlib.pyplot as plt
import numpy as np

def graficar_importancias(modelo, nombres_columnas):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importancias)), importancias[indices], align='center')
    plt.yticks(range(len(importancias)), [nombres_columnas[i] for i in indices])
    plt.title("Importancia de características - Mejor modelo ajustado")
    plt.tight_layout()
    plt.show()

def graficar_f1_scores(f1_base, f1_grid, f1_random):
    modelos = ['Base', 'Grid Search', 'Random Search']
    scores = [f1_base, f1_grid, f1_random]

    plt.figure(figsize=(6, 4))
    plt.bar(modelos, scores, color=['gray', 'blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Comparación de F1 Score entre modelos")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.show()