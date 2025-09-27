# actividad sesion 2 – clustering con dbscan y hdbscan (dataset make_moons)
#
# objetivos:
# - generar un dataset sencillo y visual (make_moons)
# - estandarizar variables (standardscaler)
# - aplicar dbscan (explorando eps y min_samples) y hdbscan (sin tuning manual)
# - evaluar con silhouette y davies-bouldin
# - visualizar resultados en 2d y guardar un resumen en csv/json/txt

import os
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# hdbscan es opcional, puede no estar instalado
# si no esta, el script sigue funcionando sin hdbscan
# instalar con: pip install hdbscan
try:
    import hdbscan
    HDBSCAN_OK = True
except Exception:
    HDBSCAN_OK = False

# reproducibilidad
SEED = 42
np.random.seed(SEED)

# ---------------------------------
# utilidades de carpeta y graficos
# ---------------------------------

def asegurar_directorio_en_script(nombre: str) -> Path:
    # crea una carpeta dentro del mismo directorio del archivo actual
    base = Path(os.path.dirname(__file__))
    out = base / nombre
    out.mkdir(parents=True, exist_ok=True)
    return out

def es_valido_para_metricas(labels: np.ndarray) -> bool:
    # requiere al menos 2 clusters distintos excluyendo ruido (-1)
    lbl = labels[labels != -1]
    if lbl.size == 0:
        return False
    return np.unique(lbl).size >= 2

def calcular_metricas(x: np.ndarray, labels: np.ndarray):
    # calcula silhouette y davies-bouldin si hay al menos 2 clusters
    if es_valido_para_metricas(labels):
        sil = float(silhouette_score(x, labels, metric="euclidean"))
        dbi = float(davies_bouldin_score(x, labels))
        return sil, dbi
    return None, None

def graficar_clusters(x2: np.ndarray, labels: np.ndarray, titulo: str, outpath: Path):
    # grafica en 2d las etiquetas; el ruido (-1) se pinta en gris claro
    plt.figure(figsize=(6, 5))
    ruido = labels == -1
    # ruido
    if ruido.any():
        plt.scatter(x2[ruido, 0], x2[ruido, 1], s=20, c="lightgray", label="ruido (-1)")
    # clusters
    resto = ~ruido
    if resto.any():
        unicos = np.unique(labels[resto])
        for c in unicos:
            idx = labels == c
            plt.scatter(x2[idx, 0], x2[idx, 1], s=20, label=f"cluster {int(c)}")
        plt.legend(loc="best")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# -----------------------------
# flujo principal de la sesion
# -----------------------------

def main():
    outdir = asegurar_directorio_en_script("resultados_sesion2")

    # 1) dataset sencillo y visual: make_moons
    #    se agrega ruido para que los metodos basados en densidad tengan sentido
    x, y_true = make_moons(n_samples=1000, noise=0.10, random_state=SEED)

    # 2) preprocesamiento: escalado y pca a 2d (aqui ya es 2d, pero dejo pca para mantener la estructura general)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2, random_state=SEED)
    x2 = pca.fit_transform(x_scaled)

    # 3) dbscan: pequena exploracion de hiperparametros
    grid_eps = [0.10, 0.15, 0.20, 0.25, 0.30]
    grid_min_samples = [3, 5, 10]

    resultados_dbscan = []
    mejor_db = None  # mejor combinacion segun criterio

    for eps in grid_eps:
        for ms in grid_min_samples:
            modelo = DBSCAN(eps=eps, min_samples=ms, metric="euclidean", n_jobs=-1)
            labels = modelo.fit_predict(x2)
            sil, dbi = calcular_metricas(x2, labels)

            registro = {
                "eps": eps,
                "min_samples": ms,
                "n_clusters_sin_ruido": int(np.unique(labels[labels != -1]).size),
                "silhouette": sil,
                "davies_bouldin": dbi
            }
            resultados_dbscan.append(registro)

            # seleccion del mejor: prioriza tener silhouette valido y mas alto
            # si ninguno tiene silhouette valido, usa menor davies-bouldin no-nulo
            if mejor_db is None:
                mejor_db = {"eps": eps, "min_samples": ms, "labels": labels, "sil": sil, "dbi": dbi}
            else:
                actual_tiene_sil = sil is not None
                mejor_tiene_sil = mejor_db["sil"] is not None
                if actual_tiene_sil and not mejor_tiene_sil:
                    mejor_db = {"eps": eps, "min_samples": ms, "labels": labels, "sil": sil, "dbi": dbi}
                elif actual_tiene_sil and mejor_tiene_sil and sil > mejor_db["sil"]:
                    mejor_db = {"eps": eps, "min_samples": ms, "labels": labels, "sil": sil, "dbi": dbi}
                elif not actual_tiene_sil and not mejor_tiene_sil:
                    # ninguno tiene silhouette valido, minimizo davies-bouldin si ambos no son None
                    if (dbi is not None) and (mejor_db["dbi"] is not None) and (dbi < mejor_db["dbi"]):
                        mejor_db = {"eps": eps, "min_samples": ms, "labels": labels, "sil": sil, "dbi": dbi}

    # grafico del mejor dbscan
    if mejor_db is not None:
        titulo_db = "dbscan (mejor) - eps={:.2f}, min_samples={}, sil={}, dbi={}".format(
            mejor_db["eps"], mejor_db["min_samples"],
            "n/a" if mejor_db["sil"] is None else f"{mejor_db['sil']:.3f}",
            "n/a" if mejor_db["dbi"] is None else f"{mejor_db['dbi']:.3f}"
        )
        graficar_clusters(x2, mejor_db["labels"], titulo_db, outdir / "dbscan_mejor.png")

    # guardo la grilla de dbscan en csv
    with open(outdir / "dbscan_grid_resultados.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eps", "min_samples", "n_clusters_sin_ruido", "silhouette", "davies_bouldin"])
        for r in resultados_dbscan:
            writer.writerow([r["eps"], r["min_samples"], r["n_clusters_sin_ruido"], r["silhouette"], r["davies_bouldin"]])

    # 4) hdbscan (si esta disponible)
    resultado_hdb = None
    if HDBSCAN_OK:
        # hdbscan tiende a descubrir la estructura sin ajustar eps; permito min_cluster_size moderado
        hdb = hdbscan.HDBSCAN(min_cluster_size=15, metric="euclidean")
        labels_hdb = hdb.fit_predict(x2)
        sil_hdb, dbi_hdb = calcular_metricas(x2, labels_hdb)
        resultado_hdb = {
            "min_cluster_size": 15,
            "n_clusters_sin_ruido": int(np.unique(labels_hdb[labels_hdb != -1]).size),
            "silhouette": sil_hdb,
            "davies_bouldin": dbi_hdb
        }
        titulo_hdb = "hdbscan - mcs=15, sil={}, dbi={}".format(
            "n/a" if sil_hdb is None else f"{sil_hdb:.3f}",
            "n/a" if dbi_hdb is None else f"{dbi_hdb:.3f}"
        )
        graficar_clusters(x2, labels_hdb, titulo_hdb, outdir / "hdbscan.png")
    else:
        # si no esta instalado, dejo una nota
        resultado_hdb = {
            "min_cluster_size": None,
            "n_clusters_sin_ruido": None,
            "silhouette": None,
            "davies_bouldin": None,
            "nota": "hdbscan no disponible: instalar con 'pip install hdbscan'"
        }

    # 5) comparacion y conclusion rapida
    # criterio: prefiero el que tenga silhouette valido y mas alto; si ninguno, menor davies-bouldin
    ganador = "empate/indefinido"
    detalle_ganador = ""

    sil_db = None if (mejor_db is None) else mejor_db["sil"]
    dbi_db = None if (mejor_db is None) else mejor_db["dbi"]
    sil_hb = resultado_hdb["silhouette"]
    dbi_hb = resultado_hdb["davies_bouldin"]

    if (sil_db is not None) or (sil_hb is not None):
        # comparo solo entre los que si tienen silhouette valido
        candidatos = []
        if sil_db is not None:
            candidatos.append(("dbscan", sil_db))
        if sil_hb is not None:
            candidatos.append(("hdbscan", sil_hb))
        if len(candidatos) > 0:
            ganador = max(candidatos, key=lambda t: t[1])[0]
    else:
        # nadie tiene silhouette valido, uso davies-bouldin (menor es mejor) si ambos no son None
        if (dbi_db is not None) and (dbi_hb is not None):
            ganador = "dbscan" if dbi_db < dbi_hb else "hdbscan"
        elif dbi_db is not None:
            ganador = "dbscan"
        elif dbi_hb is not None:
            ganador = "hdbscan"

    if ganador == "dbscan" and mejor_db is not None:
        detalle_ganador = f"dbscan con eps={mejor_db['eps']}, min_samples={mejor_db['min_samples']}"
    elif ganador == "hdbscan":
        detalle_ganador = "hdbscan con min_cluster_size=15"
    else:
        detalle_ganador = "no hay diferencias claras segun las metricas disponibles"

    # 6) persistencia de resultados (json + txt)
    resumen = {
        "dataset": "make_moons (n=1000, noise=0.10)",
        "preprocesamiento": {"scaler": "standardscaler", "pca": 2},
        "dbscan_mejor": {
            "eps": None if mejor_db is None else mejor_db["eps"],
            "min_samples": None if mejor_db is None else mejor_db["min_samples"],
            "silhouette": None if mejor_db is None else mejor_db["sil"],
            "davies_bouldin": None if mejor_db is None else mejor_db["dbi"],
        },
        "hdbscan": resultado_hdb,
        "criterio_ganador": "silhouette valido mas alto; si no, menor davies-bouldin",
        "ganador": ganador,
        "detalle_ganador": detalle_ganador
    }

    with open(outdir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    with open(outdir / "resumen.txt", "w", encoding="utf-8") as f:
        f.write("actividad sesion 2 – clustering con dbscan y hdbscan\n")
        f.write("dataset: make_moons (n=1000, noise=0.10)\n\n")
        f.write("mejor dbscan:\n")
        if mejor_db is None:
            f.write("  no se encontro configuracion destacada\n")
        else:
            f.write(f"  eps={mejor_db['eps']}, min_samples={mejor_db['min_samples']}\n")
            f.write(f"  silhouette={mejor_db['sil']}\n")
            f.write(f"  davies_bouldin={mejor_db['dbi']}\n")
        f.write("\n")
        f.write("hdbscan:\n")
        for k, v in resultado_hdb.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write(f"criterio de comparacion: {resumen['criterio_ganador']}\n")
        f.write(f"ganador: {ganador} ({detalle_ganador})\n")

    # 7) grafico base del dataset original (solo para referencia)
    plt.figure(figsize=(6, 5))
    plt.scatter(x2[:, 0], x2[:, 1], s=15)
    plt.title("dataset make_moons (pca 2d)")
    plt.tight_layout()
    plt.savefig(outdir / "dataset_make_moons.png", dpi=150)
    plt.close()

    print("listo. resultados guardados en:", outdir.resolve())

if __name__ == "__main__":
    main()