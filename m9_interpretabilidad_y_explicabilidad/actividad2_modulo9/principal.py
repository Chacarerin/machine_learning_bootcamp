# actividad módulo 9 / sesión 2 — interpretando modelos de texto con LIME
#
# objetivos:
# - cargar un dataset de opiniones (simulado o propio)
# - entrenar un clasificador binario: TF-IDF + LogisticRegression
# - seleccionar ≥3 frases del split de test y explicar sus predicciones con **LIME**
# - guardar las explicaciones de LIME en PNG y HTML
# - producir artefactos en `resultados_sesion2/`: métricas, LIME y resumen.json
#
# notas:
# - comentarios en tercera persona y tono pedagógico.
# - se mantienen utilidades y estructura iguales a sesiones previas.

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dependencias específicas con mensajes claros
def _check_or_die():
    try:
        import sklearn  # noqa: F401
    except Exception:
        print("\nerror: falta scikit-learn. instalar con  pip install scikit-learn\n")
        raise
    try:
        import lime  # noqa: F401
        import lime.lime_text  # noqa: F401
    except Exception:
        print("\nerror: falta LIME. instalar con  pip install lime\n")
        raise

_check_or_die()

# imports ahora que las dependencias están validadas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer


# -----------------------------
# utilidades de carpeta/semilla
# -----------------------------
def asegurar_dir_en_script(nombre: str) -> str:
    # crea una carpeta al lado del script, si no existe
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, nombre)
    os.makedirs(out, exist_ok=True)
    return out

def ruta_resultado(nombre: str, outdir: str) -> str:
    return os.path.join(outdir, nombre)

def fijar_semillas(seed: int = 42):
    np.random.seed(seed)


# -----------------------------
# datos: simulados o externos
# -----------------------------
def dataset_simulado() -> Tuple[List[str], List[int]]:
    # 0 = negativo, 1 = positivo
    textos = [
        "La atención fue pésima y no resolvieron mi problema.",
        "Excelente servicio, volvería sin dudarlo.",
        "El personal fue amable pero tardaron demasiado.",
        "Experiencia muy mala, me cobraron de más.",
        "Todo rápido y claro, salí satisfecho.",
        "Instalaciones sucias y poca organización.",
        "Buena atención y tratamiento efectivo.",
        "No volveré, trato impersonal y largo tiempo de espera.",
        "Proceso simple y eficiente, recomendado.",
        "Demasiado ruido y desorden, me fui molesto.",
        "El médico explicó con claridad y empatía.",
        "Poca información y confusión en recepción.",
        "Súper contento con el resultado final.",
        "Perdieron mis exámenes, muy frustrante.",
        "Atención correcta, aunque un poco lenta."
    ]
    y = [0,1,1,0,1,0,1,0,1,0,1,0,1,0,1]
    return textos, y

def cargar_csv(path: str, col_texto: str = "texto", col_label: str = "label") -> Tuple[List[str], List[int]]:
    # CSV con columnas 'texto' y 'label' (0/1 o negativo/positivo)
    df = pd.read_csv(path)
    if col_texto not in df.columns or col_label not in df.columns:
        raise ValueError(f"columnas requeridas no encontradas: '{col_texto}', '{col_label}'")
    xs = df[col_texto].astype(str).fillna("").tolist()
    ys = [_label_to_int(v) for v in df[col_label].tolist()]
    return xs, ys

def cargar_txt(path: str) -> Tuple[List[str], List[int]]:
    """
    TXT con formato 'texto<sep>label' por línea, donde <sep> ∈ {',',';','\\t'}.
    Etiquetas aceptadas: 0/1 o negativo/positivo.
    """
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            for sep in ["\t", ";", ","]:
                if sep in l:
                    a, b = l.split(sep, 1)
                    xs.append(a.strip())
                    ys.append(_label_to_int(b.strip()))
                    break
    return xs, ys

def _label_to_int(v) -> int:
    s = str(v).strip().lower()
    if s in {"1", "pos", "positivo", "positive"}:
        return 1
    if s in {"0", "neg", "negativo", "negative"}:
        return 0
    try:
        n = int(float(s))
        return 1 if n >= 1 else 0
    except Exception:
        return 0


# -----------------------------
# modelo: TF-IDF + LogisticRegression
# -----------------------------
def construir_pipeline(C: float = 2.0, ngram_max: int = 2) -> Pipeline:
    # pipeline sencillo y reproducible
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), min_df=1, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced"))
    ])
    return pipe


# -----------------------------
# visualización de matriz de confusión (útil para el notebook/entregable)
# -----------------------------
def plot_confusion(cm: np.ndarray, clases: List[str], outpath: str, titulo="matriz de confusión"):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(clases))); ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases); ax.set_yticklabels(clases)
    ax.set_xlabel("predicho"); ax.set_ylabel("verdadero")
    ax.set_title(titulo)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# LIME: explicación por instancia
# -----------------------------
def explicar_con_lime(pipe: Pipeline, textos: List[str], indices: List[int], outdir: str, k_features: int = 10):
    """
    Genera explicaciones LIME para los índices dados. Guarda PNG y HTML por cada ejemplo.
    """
    class_names = ["negativo", "positivo"]
    explainer = LimeTextExplainer(class_names=class_names)
    for i in indices:
        exp = explainer.explain_instance(
            textos[i],
            pipe.predict_proba,
            num_features=k_features,
            labels=[0, 1]
        )
        # PNG (para clase positiva por claridad)
        fig = exp.as_pyplot_figure(label=1)
        plt.title(f"LIME — doc {i} (clase=positivo)")
        plt.tight_layout()
        plt.savefig(ruta_resultado(f"lime_doc_{i}.png", outdir), dpi=150, bbox_inches="tight")
        plt.close()
        # HTML interactivo
        exp.save_to_file(ruta_resultado(f"lime_doc_{i}.html", outdir))


# -----------------------------
# flujo principal
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="sesión 9.2 — LIME para explicaciones locales en texto")
    parser.add_argument("--csv", type=str, default=None, help="CSV con columnas 'texto' y 'label'")
    parser.add_argument("--txt", type=str, default=None, help="TXT con 'texto<sep>label' (sep en {, ; tab})")
    parser.add_argument("--col_texto", type=str, default="texto")
    parser.add_argument("--col_label", type=str, default="label")
    parser.add_argument("--C", type=float, default=2.0, help="parámetro C de LogisticRegression")
    parser.add_argument("--ngram_max", type=int, default=2, help="n-gramas para TF-IDF (1..N)")
    parser.add_argument("--exp_indices", type=str, default="0,1,2", help="índices de test a explicar, ej. '0,2,5'")
    parser.add_argument("--k_features", type=int, default=10, help="cantidad de términos mostrados por LIME")
    args = parser.parse_args()

    fijar_semillas(42)
    outdir = asegurar_dir_en_script("resultados_sesion2")

    # 1) datos
    if args.csv and os.path.exists(args.csv):
        X, y = cargar_csv(args.csv, args.col_texto, args.col_label)
    elif args.txt and os.path.exists(args.txt):
        X, y = cargar_txt(args.txt)
    else:
        X, y = dataset_simulado()

    # split estratificado
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2) modelo
    pipe = construir_pipeline(C=args.C, ngram_max=args.ngram_max)
    pipe.fit(X_tr, y_tr)

    # 3) evaluación breve (para contexto)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    rep = classification_report(y_te, y_pred, target_names=["negativo", "positivo"], digits=4)
    cm = confusion_matrix(y_te, y_pred)
    with open(ruta_resultado("reporte_clasificacion.txt", outdir), "w", encoding="utf-8") as f:
        f.write(rep)
    plot_confusion(cm, ["negativo", "positivo"], ruta_resultado("matriz_confusion.png", outdir))

    # 4) explicaciones con LIME (≥3 ejemplos)
    total_test = len(X_te)
    exp_idx = []
    for tok in args.exp_indices.split(","):
        tok = tok.strip()
        if tok.isdigit():
            v = int(tok)
            if 0 <= v < total_test:
                exp_idx.append(v)
    if not exp_idx:
        exp_idx = list(range(min(3, total_test)))

    explicar_con_lime(pipe, X_te, exp_idx, outdir, k_features=args.k_features)

    # 5) resumen de la sesión
    resumen = {
        "n_total": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy_test": float(acc),
        "explicados": exp_idx,
        "modelo": f"TF-IDF(1..{args.ngram_max}) + LogisticRegression(C={args.C})",
        "lime_num_features": int(args.k_features),
        "notas": "Imágenes y HTML de LIME almacenados como lime_doc_*.png / lime_doc_*.html"
    }
    with open(ruta_resultado("resumen.json", outdir), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # 6) salida amigable por consola
    print("\n== evaluación ==")
    print(f"accuracy (test): {acc:.4f}")
    print("\n== ejemplos explicados ==")
    for i in exp_idx:
        print(f"[doc {i}] y_true={y_te[i]}  pred={int(y_pred[i])} :: {X_te[i][:110]}")
    print("\nresultados en:", outdir)
    print("- reporte_clasificacion.txt")
    print("- matriz_confusion.png")
    print("- lime_doc_*.png / lime_doc_*.html")
    print("- resumen.json\n")


if __name__ == "__main__":
    main()