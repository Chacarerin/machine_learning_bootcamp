# actividad sesion 1 - clustering jerarquico y reduccion de dimensionalidad (iris y wine)
# codigo en un solo archivo, comentarios en minusculas y pasos claros
#
# objetivos:
# - cargar datasets clasicos (iris, wine)
# - estandarizar variables y construir dendrogramas con linkage ward
# - obtener clusters con agglomerative clustering (k=2 y k=3, linkage ward)
# - proyectar datos a 2d con pca y t-sne para visualizar los clusters
# - guardar todas las figuras en una carpeta "resultados_sesion1" junto al script
#
# notas:
# - se fija semilla para reproducibilidad en t-sne
# - dependencias: numpy, pandas, matplotlib, scikit-learn, scipy

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

# reproducibilidad
SEED = 42
np.random.seed(SEED)

def asegurar_directorio_en_script(nombre: str) -> Path:
    # crea una carpeta dentro del mismo directorio del archivo actual
    base = Path(os.path.dirname(__file__))
    out = base / nombre
    out.mkdir(parents=True, exist_ok=True)
    return out

def cargar_dataset(nombre: str):
    # carga iris o wine desde sklearn y devuelve X (np.ndarray), y (np.ndarray), feature_names (list)
    if nombre == "iris":
        d = datasets.load_iris()
    elif nombre == "wine":
        d = datasets.load_wine()
    else:
        raise ValueError("dataset no soportado (use 'iris' o 'wine')")
    X = d.data.astype(float)
    y = d.target.astype(int)
    feature_names = d.feature_names
    return X, y, feature_names

def graficar_dendrograma(X_std, titulo, outpath: Path):
    # calcula linkage ward y grafica dendrograma
    Z = linkage(X_std, method="ward")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, ax=ax, color_threshold=None, no_labels=True)
    ax.set_title(titulo)
    ax.set_xlabel("observaciones"); ax.set_ylabel("distancia (ward)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def visualizar_proyecciones(X_std, labels, titulo_base, outdir: Path):
    # genera scatter de pca (2d) y tsne (2d) coloreado por etiquetas de cluster
    # pca
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_std)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=35, alpha=0.9)
    ax.set_title(f"{titulo_base} - pca (2d)")
    ax.set_xlabel("pc1"); ax.set_ylabel("pc2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(outdir / f"clusters_pca_{titulo_base.lower()}.png", dpi=150)
    plt.close(fig)

    # tsne
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X_std)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, s=35, alpha=0.9)
    ax.set_title(f"{titulo_base} - t-sne (2d)")
    ax.set_xlabel("tsne1"); ax.set_ylabel("tsne2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(outdir / f"clusters_tsne_{titulo_base.lower()}.png", dpi=150)
    plt.close(fig)

def ejecutar_para_dataset(nombre_ds: str, outdir: Path):
    # 1) carga
    X, y, feat = cargar_dataset(nombre_ds)

    # 2) estandarizacion
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 3) dendrograma ward
    graficar_dendrograma(X_std, f"dendrograma - {nombre_ds}", outdir / f"dendrograma_{nombre_ds}.png")

    # 4) clustering jerarquico (k=2 y k=3) usando ward
    # scikit-learn >= 1.2: eliminar 'affinity' y 'metric' no es necesario con linkage='ward' (usa euclidiana)
    clust2 = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels2 = clust2.fit_predict(X_std)

    clust3 = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels3 = clust3.fit_predict(X_std)

    # 5) visualizaciones de proyecciones para k=3
    visualizar_proyecciones(X_std, labels3, nombre_ds, outdir)

    # 6) guardar resumen simple a json
    resumen = {
        "dataset": nombre_ds,
        "n_observaciones": int(X.shape[0]),
        "n_variables": int(X.shape[1]),
        "clusters_k2_conteo": {int(k): int(v) for k, v in zip(*np.unique(labels2, return_counts=True))},
        "clusters_k3_conteo": {int(k): int(v) for k, v in zip(*np.unique(labels3, return_counts=True))},
        "nota": "las visualizaciones pca/tsne corresponden a k=3 con linkage ward"
    }
    (outdir / f"resumen_{nombre_ds}.json").write_text(__import__("json").dumps(resumen, indent=2), encoding="utf-8")

def main():
    outdir = asegurar_directorio_en_script("resultados_sesion1")
    for ds in ["iris", "wine"]:
        ejecutar_para_dataset(ds, outdir)
    print("listo. resultados en:", outdir.resolve())
    print("archivos:")
    print(" - dendrograma_iris.png, clusters_pca_iris.png, clusters_tsne_iris.png")
    print(" - dendrograma_wine.png, clusters_pca_wine.png, clusters_tsne_wine.png")
    print(" - resumen_iris.json, resumen_wine.json")

if __name__ == "__main__":
    main()
