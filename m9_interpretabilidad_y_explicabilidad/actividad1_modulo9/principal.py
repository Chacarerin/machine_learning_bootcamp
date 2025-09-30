# actividad módulo 9 / sesión 1 — interpretando modelos de clasificación de opiniones con LIME y SHAP
#
# objetivos:
# - construir o cargar un dataset pequeño de opiniones etiquetadas (positivo/negativo)
# - entrenar un clasificador binario simple de texto (TF-IDF + LogisticRegression por defecto)
# - aplicar **LIME** para explicar al menos 2–3 instancias y guardar visualizaciones (PNG/HTML)
# - aplicar **SHAP** para las mismas instancias y guardar visualizaciones (bar/waterfall + HTML)
# - comparar explicaciones LIME vs SHAP y guardar una síntesis de acuerdo/desacuerdo

import os
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# dependencia con mensajes útiles
# -----------------------------
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
    try:
        import shap  # noqa: F401
    except Exception:
        print("\nerror: falta SHAP. instalar con  pip install shap\n")
        raise

_check_or_die()

# ahora sí se importan los módulos
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from lime.lime_text import LimeTextExplainer
import shap

# -----------------------------
# utilidades de carpeta/semilla
# -----------------------------
def asegurar_dir_en_script(nombre: str) -> str:
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

def cargar_txt(path: str) -> Tuple[List[str], List[int]]:
    """
    TXT con 'texto<TAB/;>,label' por línea o dos columnas separadas por coma.
    Acepta separadores: ',', ';' o tab. Etiquetas esperadas: 0/1 o negativo/positivo.
    """
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            # intenta varios separadores
            for sep in ["\t", ";", ","]:
                if sep in l:
                    a, b = l.split(sep, 1)
                    xs.append(a.strip())
                    ys.append(_label_to_int(b.strip()))
                    break
            else:
                # si no hay separador, asume solo texto y etiqueta faltante (descarta)
                pass
    return xs, ys

def cargar_csv(path: str, col_texto: str = "texto", col_label: str = "label") -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)
    if col_texto not in df.columns or col_label not in df.columns:
        raise ValueError(f"columnas requeridas no encontradas: '{col_texto}', '{col_label}'")
    xs = df[col_texto].astype(str).fillna("").tolist()
    ys = [_label_to_int(v) for v in df[col_label].tolist()]
    return xs, ys

def _label_to_int(v) -> int:
    s = str(v).strip().lower()
    if s in {"1", "pos", "positivo", "positive"}:
        return 1
    if s in {"0", "neg", "negativo", "negative"}:
        return 0
    # fallback
    try:
        n = int(float(s))
        return 1 if n >= 1 else 0
    except Exception:
        return 0

# -----------------------------
# modelo base (pipeline)
# -----------------------------
def construir_pipeline(C: float = 2.0, ngram_max: int = 2) -> Pipeline:
    """
    Clasificador sencillo: TF-IDF (uni/bi-gramas) + LogisticRegression.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), min_df=1, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced"))
    ])
    return pipe

# -----------------------------
# gráficos auxiliares
# -----------------------------
def plot_confusion(cm: np.ndarray, clases: List[str], outpath: str, titulo="matriz de confusión"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(clases))); ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases); ax.set_yticklabels(clases)
    ax.set_xlabel("predicho"); ax.set_ylabel("verdadero")
    ax.set_title(titulo)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# LIME
# -----------------------------
def explicar_con_lime(pipe: Pipeline, textos: List[str], indices: List[int], outdir: str):
    """
    Genera explicaciones LIME para los índices indicados. Guarda PNG y HTML.
    """
    class_names = ["negativo", "positivo"]
    explainer = LimeTextExplainer(class_names=class_names)
    for i in indices:
        exp = explainer.explain_instance(
            textos[i],
            pipe.predict_proba,
            num_features=10,
            labels=[0, 1]
        )
        # PNG (bar plot de contribuciones para la clase positiva por defecto)
        fig = exp.as_pyplot_figure(label=1)
        plt.title(f"LIME — doc {i} (clase=positivo)")
        out_png = ruta_resultado(f"lime_doc_{i}.png", outdir)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()

        # HTML interactivo (útil para inspección detallada)
        out_html = ruta_resultado(f"lime_doc_{i}.html", outdir)
        exp.save_to_file(out_html)

# -----------------------------
# SHAP
# -----------------------------
def explicar_con_shap(pipe: Pipeline, textos: List[str], indices: List[int], outdir: str):
    """
    Usa SHAP con masker de texto para explicar las mismas instancias.
    Se generan bar plots (top rasgos) y waterfall; además se guarda HTML de texto.
    """
    # función de probas para clase positiva
    f = lambda xs: pipe.predict_proba(list(xs))[:, 1]
    masker = shap.maskers.Text()
    explainer = shap.Explainer(f, masker)
    # textos a explicar
    docs = [textos[i] for i in indices]
    sv = explainer(docs)  # valores shap por token

    for k, i in enumerate(indices):
        # 1) bar plot (top contribuciones absolutas)
        plt.figure(figsize=(6.4, 4.2))
        shap.plots.bar(sv[k], show=False, max_display=12)
        out_bar = ruta_resultado(f"shap_bar_doc_{i}.png", outdir)
        plt.tight_layout(); plt.savefig(out_bar, dpi=150, bbox_inches="tight"); plt.close()

        # 2) waterfall (cuando aplica)
        try:
            plt.figure(figsize=(6.4, 4.2))
            shap.plots.waterfall(sv[k], show=False, max_display=12)
            out_wf = ruta_resultado(f"shap_waterfall_doc_{i}.png", outdir)
            plt.tight_layout(); plt.savefig(out_wf, dpi=150, bbox_inches="tight"); plt.close()
        except Exception:
            # algunos entornos de texto no soportan waterfall directamente; se ignora de forma segura
            pass

        # 3) HTML (visualización de texto coloreado por contribución)
        try:
            html = shap.plots.text(sv[k], display=False)
            out_html = ruta_resultado(f"shap_text_doc_{i}.html", outdir)
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html.data)
        except Exception:
            pass

# -----------------------------
# comparación simple LIME vs SHAP
# -----------------------------
def comparar_top_palabras(pipe: Pipeline, textos: List[str], idx: int, k: int = 8) -> Dict[str, List[str]]:
    """
    Obtiene top-k palabras relevantes para la clase positiva según LIME y SHAP, y calcula intersección.
    Retorna diccionario con listas para guardar en el resumen.
    """
    # LIME
    explainer = LimeTextExplainer(class_names=["negativo","positivo"])
    exp = explainer.explain_instance(textos[idx], pipe.predict_proba, num_features=20, labels=[1])
    lime_pairs = exp.as_list(label=1)
    lime_top = [w for (w, _) in lime_pairs[:k]]

    # SHAP
    f = lambda xs: pipe.predict_proba(list(xs))[:, 1]
    sv = shap.Explainer(f, shap.maskers.Text())([textos[idx]])
    # valores por token ya segmentados; se ordenan por magnitud
    shap_tokens = [str(t) for t in sv[0].data]
    shap_vals = np.abs(sv[0].values)
    order = np.argsort(shap_vals)[::-1]
    shap_top = []
    for j in order:
        tok = shap_tokens[j].strip()
        if tok and tok not in shap_top:
            shap_top.append(tok)
        if len(shap_top) >= k:
            break

    inter = sorted(list(set(lime_top) & set(shap_top)))
    return {
        "lime_top": lime_top,
        "shap_top": shap_top,
        "interseccion": inter
    }

# -----------------------------
# flujo principal
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="sesión 9.1 — explicabilidad con LIME y SHAP")
    parser.add_argument("--csv", type=str, default=None, help="ruta a CSV con columnas 'texto' y 'label'")
    parser.add_argument("--txt", type=str, default=None, help="ruta a TXT (texto<sep>label por línea; sep en {, ; tab})")
    parser.add_argument("--col_texto", type=str, default="texto")
    parser.add_argument("--col_label", type=str, default="label")
    parser.add_argument("--C", type=float, default=2.0, help="C de LogisticRegression")
    parser.add_argument("--ngram_max", type=int, default=2, help="n-gramas TF-IDF (1..N)")
    parser.add_argument("--exp_indices", type=str, default="0,1,2",
                        help="índices de test a explicar, separados por coma (p.ej. '0,2,5')")
    args = parser.parse_args()

    fijar_semillas(42)
    outdir = asegurar_dir_en_script("resultados_sesion1")

    # 1) carga de datos
    if args.csv and os.path.exists(args.csv):
        X, y = cargar_csv(args.csv, args.col_texto, args.col_label)
    elif args.txt and os.path.exists(args.txt):
        X, y = cargar_txt(args.txt)
    else:
        X, y = dataset_simulado()

    # split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2) modelo
    pipe = construir_pipeline(C=args.C, ngram_max=args.ngram_max)
    pipe.fit(X_tr, y_tr)

    # 3) evaluación básica
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    rep = classification_report(y_te, y_pred, target_names=["negativo","positivo"], digits=4)
    cm = confusion_matrix(y_te, y_pred)

    with open(ruta_resultado("reporte_clasificacion.txt", outdir), "w", encoding="utf-8") as f:
        f.write(rep)

    plot_confusion(cm, ["negativo","positivo"], ruta_resultado("matriz_confusion.png", outdir))

    # 4) explicaciones: indices seleccionados sobre el split de test
    # normaliza string "0,1,2" a lista de enteros válidos
    total_test = len(X_te)
    exp_idx = []
    for tok in args.exp_indices.split(","):
        tok = tok.strip()
        if tok.isdigit():
            v = int(tok)
            if 0 <= v < total_test:
                exp_idx.append(v)
    # si el usuario pasó índices fuera de rango o vacíos, por defecto toma los tres primeros
    if not exp_idx:
        exp_idx = list(range(min(3, total_test)))

    # LIME
    explicar_con_lime(pipe, X_te, exp_idx, outdir)
    # SHAP
    explicar_con_shap(pipe, X_te, exp_idx, outdir)

    # 5) comparación simple LIME vs SHAP (palabras top) para cada índice
    comparaciones = {}
    for i in exp_idx:
        comparaciones[f"doc_{i}"] = comparar_top_palabras(pipe, X_te, i, k=8)

    # 6) resumen
    resumen = {
        "n_total": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy_test": float(acc),
        "explicaciones_indices": exp_idx,
        "comparaciones_top": comparaciones,
        "modelo": "TF-IDF(1..{}) + LogisticRegression(C={})".format(args.ngram_max, args.C),
        "notas": "Se guardan LIME (PNG/HTML) y SHAP (bar/waterfall + HTML)."
    }
    with open(ruta_resultado("resumen.json", outdir), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # 7) consola amigable
    print("\n== evaluación ==")
    print(f"accuracy (test): {acc:.4f}")
    print("\n== ejemplos de test y predicción ==")
    for i in exp_idx:
        print(f"[doc {i}] y_true={y_te[i]} pred={int(y_pred[i])} :: {X_te[i][:110]}")

    print("\nresultados en:", outdir)
    print("- reporte_clasificacion.txt")
    print("- matriz_confusion.png")
    print("- lime_doc_*.png / .html")
    print("- shap_bar_doc_*.png / shap_waterfall_doc_*.png / shap_text_doc_*.html")
    print("- resumen.json\n")

if __name__ == "__main__":
    main()