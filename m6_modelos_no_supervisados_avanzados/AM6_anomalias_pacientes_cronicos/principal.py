# modulo 6 - segmentacion y deteccion de anomalias en pacientes cronicos

# librerias basicas
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# graficos
import matplotlib.pyplot as plt
import seaborn as sns

# reduccion y escalamiento
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # paquete se instala como umap-learn

# clustering y metricas
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan

# deteccion de anomalias
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import shutil

# semilla global para reproducibilidad
RANDOM_STATE = 42


# fija la semilla
def seed_everything(seed: int = RANDOM_STATE):
    np.random.seed(seed)


# crea una carpeta si no existe
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# guarda un dataframe a csv sin index
def save_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


# estilo grafico simple y consistente
def set_plot_style():
    sns.set(context="notebook", style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (7, 6),
        "savefig.dpi": 180,
        "axes.titlesize": 12,
        "axes.labelsize": 10
    })


# intenta descargar el dataset desde kaggle con kagglehub
def intentar_descargar_diabetes(destino: str) -> bool:
    try:
        import kagglehub
    except Exception:
        print("[aviso] instalar 'kagglehub' si se quiere descarga automatica (pip install kagglehub)")
        return False

    try:
        path = kagglehub.dataset_download("mathchi/diabetes-data-set")
        origen = os.path.join(path, "diabetes.csv")
        if os.path.exists(origen):
            ensure_dir(os.path.dirname(destino))
            shutil.copy(origen, destino)
            print(f"[ok] dataset copiado a {destino}")
            return True
        print("[error] no se encontro 'diabetes.csv' dentro del dataset descargado")
        return False
    except Exception as e:
        print(f"[error] no fue posible descargar el dataset automaticamente: {e}")
        return False


# carga el csv; si no existe, intenta descargarlo; estandariza nombres de columnas
def cargar_dataset(ruta: str = "./data/diabetes.csv") -> pd.DataFrame | None:
    if not os.path.exists(ruta):
        print(f"[aviso] no se encontro {ruta}, intentando descargar desde kaggle...")
        ok = intentar_descargar_diabetes(ruta)
        if not ok:
            return None

    df = pd.read_csv(ruta)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# limpia nulos, elimina posibles columnas objetivo y escala variables numericas
def preprocesar(df: pd.DataFrame):
    df = df.dropna().reset_index(drop=True)

    # si hubiera alguna etiqueta, se elimina (trabajo no supervisado)
    posibles_y = [c for c in df.columns if c in ("class", "outcome", "diabetes")]
    x_df = df.drop(columns=posibles_y) if posibles_y else df.copy()

    # identifica columnas numericas; si quedan categoricas, las pasa a dummies
    num_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in x_df.columns if c not in num_cols]
    if cat_cols:
        x_df = pd.get_dummies(x_df, columns=cat_cols, drop_first=True)
        num_cols = x_df.columns.tolist()

    # escalamiento tipo z-score
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df[num_cols])

    return x_df, num_cols, x_scaled, scaler


# calcula pca, t-sne y umap a 2 dimensiones para visualizacion
def reducciones(x_scaled: np.ndarray) -> dict[str, np.ndarray]:
    resultados = {}

    # pca lineal
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    resultados["pca"] = pca.fit_transform(x_scaled)

    # t-sne con perplejidad segura segun tamaño de muestra
    n = x_scaled.shape[0]
    perplejidad = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplejidad, learning_rate="auto", init="pca")
    resultados["tsne"] = tsne.fit_transform(x_scaled)

    # umap preserva vecindades locales
    um = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    resultados["umap"] = um.fit_transform(x_scaled)

    return resultados


# scatter 2d; si vienen etiquetas, colorea por cluster
def scatter_2d(z: np.ndarray, etiquetas=None, titulo: str = "proyeccion 2d", filename: str = "proyeccion.png"):
    plt.figure()
    if etiquetas is None:
        plt.scatter(z[:, 0], z[:, 1], s=18, alpha=0.85)
    else:
        n_colors = int(max(2, len(np.unique(etiquetas))))
        palette = sns.color_palette(n_colors=n_colors)
        sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=etiquetas, s=18, alpha=0.9, legend="brief", palette=palette)
        plt.legend(loc="best", fontsize=8, title="cluster")
    plt.title(titulo)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ejecuta dbscan y retorna etiquetas (-1 es ruido)
def correr_dbscan(x_scaled: np.ndarray, eps: float = 0.9, min_samples: int = 8) -> np.ndarray:
    modelo = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return modelo.fit_predict(x_scaled)


# ejecuta hdbscan y retorna etiquetas (-1 es ruido)
def correr_hdbscan(x_scaled: np.ndarray, min_cluster_size: int = 12, min_samples: int | None = None) -> np.ndarray:
    modelo = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    return modelo.fit_predict(x_scaled)


# calcula numero de clusters, porcentaje de ruido y, si aplica, silueta y davies-bouldin
def evaluar_clusters(x_scaled: np.ndarray, labels: np.ndarray, nombre: str) -> dict:
    mask_valid = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    out = {
        "modelo": nombre,
        "n_clusters": int(max(n_clusters, 0)),
        "porcentaje_ruido": float(np.mean(labels == -1) * 100.0),
        "silhouette": None,
        "davies_bouldin": None
    }

    if n_clusters >= 2 and mask_valid.sum() > 10:
        out["silhouette"] = float(silhouette_score(x_scaled[mask_valid], labels[mask_valid]))
        out["davies_bouldin"] = float(davies_bouldin_score(x_scaled[mask_valid], labels[mask_valid]))

    return out


# entrena isolation forest y retorna indices anomalos
def correr_isolation_forest(x_scaled: np.ndarray, contamination="auto"):
    iso = IsolationForest(random_state=RANDOM_STATE, contamination=contamination)
    iso.fit(x_scaled)
    y_pred = iso.predict(x_scaled)  # -1 anomalia, 1 normal
    anomalias = np.where(y_pred == -1)[0]
    return anomalias, iso


# entrena one-class svm y retorna indices anomalos
def correr_oneclass_svm(x_scaled: np.ndarray, nu: float = 0.05, gamma: str = "scale"):
    oc = OneClassSVM(nu=nu, gamma=gamma)
    oc.fit(x_scaled)
    y_pred = oc.predict(x_scaled)  # -1 anomalia, 1 normal
    anomalias = np.where(y_pred == -1)[0]
    return anomalias, oc


# compara anomalias con clusters poco frecuentes (percentil 10 de tamaños positivos)
def analisis_cruzado(labels_dict: dict[str, np.ndarray], anom_iso: np.ndarray, anom_ocsvm: np.ndarray) -> dict:
    resumen = {}

    for nombre, labels in labels_dict.items():
        tamanos = pd.Series(labels).value_counts()
        positivos = tamanos[tamanos.index != -1]

        if len(positivos) == 0:
            raros = []
            idx_raros = np.array([], dtype=int)
        else:
            umbral = np.percentile(positivos.values, 10)
            raros = positivos[positivos <= umbral].index.tolist()
            idx_raros = np.where(np.isin(labels, raros))[0]

        inter_iso = int(len(np.intersect1d(idx_raros, anom_iso)))
        inter_oc = int(len(np.intersect1d(idx_raros, anom_ocsvm)))

        resumen[nombre] = {
            "clusters_raros": raros,
            "n_obs_clusters_raros": int(len(idx_raros)),
            "interseccion_iso": inter_iso,
            "interseccion_ocsvm": inter_oc
        }

    return resumen


# punto de entrada: orquesta todo el flujo, genera tablas y graficos
def main():
    seed_everything()
    set_plot_style()
    inicio = time.time()

    df = cargar_dataset()
    if df is None or df.empty:
        print("[error] no fue posible cargar el dataset")
        return

    x_df, num_cols, x_scaled, scaler = preprocesar(df)

    proys = reducciones(x_scaled)
    scatter_2d(proys["pca"], None, "pca (2d)", "pca.png")
    scatter_2d(proys["tsne"], None, "t-sne (2d)", "tsne.png")
    scatter_2d(proys["umap"], None, "umap (2d)", "umap.png")

    labels_db = correr_dbscan(x_scaled, eps=0.9, min_samples=8)
    labels_hdb = correr_hdbscan(x_scaled, min_cluster_size=12, min_samples=None)

    eval_db = evaluar_clusters(x_scaled, labels_db, "dbscan")
    eval_hdb = evaluar_clusters(x_scaled, labels_hdb, "hdbscan")
    eval_df = pd.DataFrame([eval_db, eval_hdb])
    save_table(eval_df, "metricas_clustering.csv")

    scatter_2d(proys["pca"], labels_db, "dbscan sobre pca", "clusters_dbscan_pca.png")
    scatter_2d(proys["umap"], labels_db, "dbscan sobre umap", "clusters_dbscan_umap.png")
    scatter_2d(proys["pca"], labels_hdb, "hdbscan sobre pca", "clusters_hdbscan_pca.png")
    scatter_2d(proys["umap"], labels_hdb, "hdbscan sobre umap", "clusters_hdbscan_umap.png")

    anom_iso, _ = correr_isolation_forest(x_scaled, contamination="auto")
    anom_oc, _ = correr_oneclass_svm(x_scaled, nu=0.05, gamma="scale")

    save_table(pd.DataFrame({"idx": anom_iso}), "anomalias_isolation_forest.csv")
    save_table(pd.DataFrame({"idx": anom_oc}), "anomalias_oneclass_svm.csv")

    def plot_anomalias(z: np.ndarray, indices: np.ndarray, titulo: str, filename: str):
        # marca anomalias con una x sobre la proyeccion 2d
        plt.figure()
        plt.scatter(z[:, 0], z[:, 1], s=16, alpha=0.65)
        if len(indices):
            plt.scatter(z[indices, 0], z[indices, 1], s=42, alpha=0.95, marker="x")
        plt.title(titulo)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_anomalias(proys["pca"], anom_iso, "anomalias (isolation forest) en pca", "anomalias_iso_pca.png")
    plot_anomalias(proys["umap"], anom_iso, "anomalias (isolation forest) en umap", "anomalias_iso_umap.png")
    plot_anomalias(proys["pca"], anom_oc, "anomalias (one-class svm) en pca", "anomalias_ocsvm_pca.png")
    plot_anomalias(proys["umap"], anom_oc, "anomalias (one-class svm) en umap", "anomalias_ocsvm_umap.png")

    resumen = analisis_cruzado({"dbscan": labels_db, "hdbscan": labels_hdb}, anom_iso, anom_oc)
    resumen_df = pd.DataFrame(resumen)
    save_table(resumen_df, "analisis_cruzado.csv")

    fin = time.time()
    print("---- resumen ejecucion ----")
    print(eval_df)
    print(f"tiempo total de ejecucion: {fin - inicio:.2f} segundos")
    print("archivos generados:")
    print(" - pca.png, tsne.png, umap.png")
    print(" - clusters_dbscan_*.png, clusters_hdbscan_*.png")
    print(" - anomalias_*_*.png")
    print(" - metricas_clustering.csv, anomalias_isolation_forest.csv, anomalias_oneclass_svm.csv, analisis_cruzado.csv")


if __name__ == "__main__":
    main()