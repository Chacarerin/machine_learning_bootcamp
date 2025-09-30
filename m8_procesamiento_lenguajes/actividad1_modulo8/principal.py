# actividad módulo 8 / sesión 1 — nlp tradicional sobre textos clínicos breves
#
# objetivos:
# - construir o cargar un pequeño corpus (5–10+) de notas clínicas simuladas
# - limpieza básica: pasar a minúsculas y eliminar puntuación
# - representar documentos con **bag of words (CountVectorizer)** y **TF-IDF**
# - calcular **similaridad coseno** entre documentos
# - visualizar la similaridad en un **heatmap** y comentar documentos más relacionados
# - extraer y graficar los **10 términos más relevantes** en al menos 3 documentos
# - guardar artefactos (gráficos, matrices y resúmenes) en `resultados_sesion1/`

import os
import re
import json
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    print("\nerror: falta scikit-learn. instalar con: pip install scikit-learn\n")
    raise

# -----------------------------
# utilidades
# -----------------------------

def asegurar_dir_en_script(nombre: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, nombre)
    os.makedirs(out, exist_ok=True)
    return out

def ruta_resultado(nombre: str, outdir: str) -> str:
    return os.path.join(outdir, nombre)

def limpiar_basico(texto: str) -> str:
    t = texto.lower()
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def graficar_heatmap_sim(sim: np.ndarray, etiquetas: List[str], outpath: str, titulo="similaridad coseno"):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="viridis")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(etiquetas)))
    ax.set_yticks(np.arange(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax.set_yticklabels(etiquetas)
    ax.set_title(titulo)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    color="white" if sim[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def graficar_top_terms(indices: List[int], tfidf_matrix: np.ndarray, feature_names: List[str], outdir: str, k: int = 10):
    for idx in indices:
        fila = tfidf_matrix[idx, :].toarray().ravel()
        top_ids = np.argsort(fila)[-k:][::-1]
        terms = [feature_names[i] for i in top_ids]
        vals = fila[top_ids]
        plt.figure(figsize=(10, 4))
        plt.bar(range(k), vals)
        plt.xticks(range(k), terms, rotation=45, ha="right")
        plt.title(f"doc {idx}: top {k} términos TF-IDF")
        plt.ylabel("tf-idf")
        plt.tight_layout()
        plt.savefig(ruta_resultado(f"top_terminos_doc_{idx}.png", outdir), dpi=150)
        plt.close()

# -----------------------------
# corpus
# -----------------------------

def cargar_corpus_desde_txt(path: str) -> List[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                docs.append(l)
    return docs

def corpus_por_defecto() -> List[str]:
    return [
        "Paciente masculino, 45 años, presenta fiebre leve y congestión nasal. Se sospecha infección viral.",
        "Paciente femenina, 32 años, dolor abdominal persistente, sin fiebre, con historial de gastritis.",
        "Varón 60 años, tos seca, disnea leve, saturación 93%, se indica radiografía de tórax y pruebas covid.",
        "Mujer 25 años, cefalea intensa y fotofobia; se sugiere hidratación y control neurológico.",
        "Paciente 70 años con hipertensión y diabetes, control irregular de glicemia, fatiga y poliuria.",
        "Hombre 38 años, dolor torácico no irradiado, ECG sin cambios, se descarta síndrome coronario agudo.",
        "Niña 7 años, otalgia derecha, membrana timpánica eritematosa; se inicia amoxicilina.",
        "Mujer 55 años, dolor lumbar mecánico, sin déficit neurológico, se indica reposo y AINEs.",
        "Paciente 42 años, náuseas y vómitos postprandiales, prueba positiva para Helicobacter pylori.",
        "Hombre 29 años, esguince de tobillo tras actividad deportiva, edema y dolor a la palpación."
    ]

# -----------------------------
# vectorización y similitud
# -----------------------------

def vectorizar(docs_limpios: List[str], ngram_max: int = 2):
    count_vec = CountVectorizer(ngram_range=(1, ngram_max))
    tfidf_vec = TfidfVectorizer(ngram_range=(1, ngram_max))
    bow = count_vec.fit_transform(docs_limpios)
    tfidf = tfidf_vec.fit_transform(docs_limpios)
    feats = tfidf_vec.get_feature_names_out().tolist()
    return count_vec, tfidf_vec, bow, tfidf, feats

def matriz_similaridad(tfidf):
    return cosine_similarity(tfidf)

# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="sesión 8.1 — NLP tradicional en textos clínicos")
    parser.add_argument("--corpus_txt", type=str, default=None,
                        help="ruta a .txt con un documento por línea; si se omite, se usa un corpus de ejemplo")
    parser.add_argument("--ngram_max", type=int, default=2, help="máximo tamaño de n-grama")
    parser.add_argument("--k_top", type=int, default=10, help="cantidad de términos a mostrar por documento")
    args = parser.parse_args()

    outdir = asegurar_dir_en_script("resultados_sesion1")

    if args.corpus_txt and os.path.exists(args.corpus_txt):
        docs_raw = cargar_corpus_desde_txt(args.corpus_txt)
    else:
        docs_raw = corpus_por_defecto()

    etiquetas = [f"doc{i}" for i in range(len(docs_raw))]
    docs_clean = [limpiar_basico(t) for t in docs_raw]

    with open(ruta_resultado("corpus_limpio.json", outdir), "w", encoding="utf-8") as f:
        json.dump({"docs": docs_clean}, f, ensure_ascii=False, indent=2)

    count_vec, tfidf_vec, bow_mat, tfidf_mat, feats = vectorizar(docs_clean, ngram_max=args.ngram_max)
    sim = matriz_similaridad(tfidf_mat)

    graficar_heatmap_sim(sim, etiquetas, ruta_resultado("heatmap_similaridad.png", outdir),
                         titulo="similaridad coseno entre notas clínicas")

    idx_docs = list(range(min(3, len(docs_clean))))
    graficar_top_terms(idx_docs, tfidf_mat, feats, outdir, k=args.k_top)

    relacionados = {}
    for i in range(sim.shape[0]):
        orden = np.argsort(sim[i])[::-1]
        orden = [j for j in orden if j != i]
        relacionados[etiquetas[i]] = {"mas_relacionado": etiquetas[orden[0]], "sim": float(sim[i, orden[0]])}

    np.savetxt(ruta_resultado("bow_matrix_shape.txt", outdir), np.array(bow_mat.shape)[None, :], fmt="%d")
    np.savetxt(ruta_resultado("tfidf_matrix_shape.txt", outdir), np.array(tfidf_mat.shape)[None, :], fmt="%d")
    np.savetxt(ruta_resultado("similaridad_coseno_matrix.txt", outdir), sim, fmt="%.6f")

    resumen = {
        "n_docs": len(docs_clean),
        "ngram_max": args.ngram_max,
        "k_top": args.k_top,
        "documentos_mas_relacionados": relacionados,
        "nota": "las figuras 'heatmap_similaridad.png' y 'top_terminos_doc_*.png' ilustran similitudes y términos clave."
    }
    with open(ruta_resultado("resumen.json", outdir), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("\nresultados en:", outdir)
    print("- heatmap_similaridad.png")
    for i in idx_docs:
        print(f"- top_terminos_doc_{i}.png")
    print("- similaridad_coseno_matrix.txt")
    print("- corpus_limpio.json")
    print("- resumen.json\n")

if __name__ == "__main__":
    main()