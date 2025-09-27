# actividad sesion 3 – reduccion de dimensionalidad con pca (dataset iris)
#
# objetivos:
# - cargar un dataset multivariado (iris)
# - estandarizar variables (standardscaler)
# - aplicar pca a 2 componentes
# - graficar varianza explicada (individual y acumulada)
# - visualizar la proyeccion 2d coloreando por clase
# - guardar un resumen en csv/json/txt manteniendo el mismo estilo de trabajo
# - (opcional) comparar knn con y sin pca para ver impacto simple en clasificacion

import os
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def graficar_varianza_explicada(pca_obj, outpath: Path):
    # grafica barras de varianza individual y linea de varianza acumulada
    var_ind = pca_obj.explained_variance_ratio_
    var_acum = np.cumsum(var_ind)
    idx = np.arange(1, len(var_ind) + 1)

    plt.figure(figsize=(6, 4))
    plt.bar(idx, var_ind, label="varianza (individual)")
    plt.plot(idx, var_acum, marker="o", label="varianza (acumulada)")
    plt.xticks(idx)
    plt.xlabel("componente principal")
    plt.ylabel("proporcion de varianza")
    plt.title("pca – varianza explicada")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def graficar_pca_2d(x2: np.ndarray, y: np.ndarray, nombres_clase, outpath: Path):
    # grafica la proyeccion 2d de pca con colores por clase
    plt.figure(figsize=(6, 5))
    clases = np.unique(y)
    for c in clases:
        m = y == c
        plt.scatter(x2[m, 0], x2[m, 1], s=22, alpha=0.9, label=nombres_clase[c])
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.title("proyeccion 2d con pca (iris)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------------------------------
# helpers de persistencia de texto
# ---------------------------------

def escribir_txt(texto: str, outpath: Path):
    # guarda texto plano
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(texto)

def escribir_json(data: dict, outpath: Path):
    # guarda json con indentado legible
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def guardar_scores_csv(x2: np.ndarray, y: np.ndarray, outpath: Path):
    # guarda un csv con pc1, pc2 y la clase
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pc1", "pc2", "clase"])
        for i in range(x2.shape[0]):
            w.writerow([float(x2[i, 0]), float(x2[i, 1]), int(y[i])])

def guardar_componentes_csv(pca_obj, nombres_variables, outpath: Path):
    # guarda la matriz de cargas (componentes) del pca
    # filas: pc1, pc2 ; columnas: variables originales
    comps = pca_obj.components_
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["componente"] + list(nombres_variables))
        for i, fila in enumerate(comps, start=1):
            w.writerow([f"pc{i}"] + [float(v) for v in fila])

# ---------------------------------
# flujo principal de la sesion
# ---------------------------------

def main():
    outdir = asegurar_directorio_en_script("resultados_sesion3")

    # 1) dataset multivariado: iris
    datos = load_iris()
    x = datos.data
    y = datos.target
    nombres_clase = datos.target_names
    nombres_variables = datos.feature_names

    # 2) preprocesamiento: escalado y pca a 2d
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2, random_state=SEED)
    x2 = pca.fit_transform(x_scaled)

    # 3) graficos principales
    graficar_varianza_explicada(pca, outdir / "01_varianza_explicada.png")
    graficar_pca_2d(x2, y, nombres_clase, outdir / "02_pca_2d.png")

    # 4) persistencia de artefactos (csv + json + txt)
    guardar_scores_csv(x2, y, outdir / "pca_scores_pc12.csv")
    guardar_componentes_csv(pca, nombres_variables, outdir / "pca_componentes_pc12.csv")

    var_ind = pca.explained_variance_ratio_
    var_acum = float(np.cumsum(var_ind)[-1])

    resumen = {
        "dataset": "iris",
        "n_observaciones": int(x.shape[0]),
        "n_variables": int(x.shape[1]),
        "preprocesamiento": {"scaler": "standardscaler", "pca_componentes": 2},
        "varianza_explicada": {
            "pc1": float(var_ind[0]),
            "pc2": float(var_ind[1]),
            "acumulada_pc1_pc2": float(var_acum)
        },
        "archivos_generados": {
            "png_varianza": "01_varianza_explicada.png",
            "png_pca_2d": "02_pca_2d.png",
            "csv_scores": "pca_scores_pc12.csv",
            "csv_componentes": "pca_componentes_pc12.csv",
            "txt_interpretacion": "resumen.txt",
            "json_resumen": "resumen.json",
            "txt_metricas_knn": "metricas_knn.txt"
        },
        "nota": "en iris, dos componentes suelen capturar una fraccion alta de la variabilidad permitiendo visualizar separacion parcial entre clases."
    }
    escribir_json(resumen, outdir / "resumen.json")

    texto = []
    texto.append("actividad sesion 3 – reduccion de dimensionalidad con pca (iris)\n")
    texto.append(f"observaciones: {x.shape[0]} | variables: {x.shape[1]}\n")
    texto.append("preprocesamiento: standardscaler -> pca (2 componentes)\n")
    texto.append(f"varianza pc1: {var_ind[0]:.4f} | varianza pc2: {var_ind[1]:.4f} | acumulada: {var_acum:.4f}\n")
    texto.append("comentario: la proyeccion 2d muestra patrones consistentes con las clases de iris; "
                 "pca ayuda a visualizar estructura y a reducir ruido para analisis exploratorio.\n")
    escribir_txt("\n".join(texto), outdir / "resumen.txt")

    # 5) (opcional) clasificacion simple: knn con y sin pca
    #    mantiene el mismo estilo de comparacion breve que usamos antes
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED
    )

    pipe_sin_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    pipe_con_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=SEED)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])

    pipe_sin_pca.fit(x_tr, y_tr)
    pipe_con_pca.fit(x_tr, y_tr)

    y_pred_sin = pipe_sin_pca.predict(x_te)
    y_pred_con = pipe_con_pca.predict(x_te)

    acc_sin = accuracy_score(y_te, y_pred_sin)
    acc_con = accuracy_score(y_te, y_pred_con)

    cv_sin = cross_val_score(pipe_sin_pca, x, y, cv=5).mean()
    cv_con = cross_val_score(pipe_con_pca, x, y, cv=5).mean()

    rep_sin = classification_report(y_te, y_pred_sin, zero_division=0)
    rep_con = classification_report(y_te, y_pred_con, zero_division=0)
    cm_sin = confusion_matrix(y_te, y_pred_sin)
    cm_con = confusion_matrix(y_te, y_pred_con)

    lineas = []
    lineas.append("# comparacion knn sin pca vs con pca (2 componentes)")
    lineas.append(f"accuracy test sin pca: {acc_sin:.4f}")
    lineas.append(f"accuracy test con pca: {acc_con:.4f}")
    lineas.append(f"cv(5) promedio sin pca: {cv_sin:.4f}")
    lineas.append(f"cv(5) promedio con pca: {cv_con:.4f}")
    lineas.append("")
    lineas.append("reporte clasificacion sin pca:")
    lineas.append(rep_sin)
    lineas.append("reporte clasificacion con pca:")
    lineas.append(rep_con)
    lineas.append("matriz de confusion sin pca:")
    lineas.append(str(cm_sin))
    lineas.append("matriz de confusion con pca:")
    lineas.append(str(cm_con))

    escribir_txt("\n".join(lineas), outdir / "metricas_knn.txt")

    print("listo. resultados guardados en:", outdir.resolve())
    print("- 01_varianza_explicada.png")
    print("- 02_pca_2d.png")
    print("- pca_scores_pc12.csv")
    print("- pca_componentes_pc12.csv")
    print("- resumen.json")
    print("- resumen.txt")
    print("- metricas_knn.txt")

if __name__ == "__main__":
    main()