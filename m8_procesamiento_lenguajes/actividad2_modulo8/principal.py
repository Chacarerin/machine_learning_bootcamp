# actividad módulo 8 / sesión 2 — preprocesamiento de texto (spaCy + NLTK) y TF-IDF en notas clínicas
#
# objetivos:
# - simular o cargar un corpus breve (≥10) de notas clínicas
# - limpieza básica: minúsculas, remover signos, números, correos y URLs
# - tokenización + lematización con **spaCy** (con fallback a NLTK/stemming si spaCy no está disponible)
# - eliminación de stopwords (spaCy o NLTK)
# - vectorización con **TfidfVectorizer** (n-gramas hasta bi-gramas)
# - visualización de **términos más relevantes por documento**
# - comparación corpus **original vs preprocesado**: longitud media, tamaño de vocabulario, repetición (TTR)
# - guardar artefactos en `resultados_sesion2/` (figuras, matrices, resúmenes)

import os
import re
import json
import argparse
from typing import List, Tuple, Iterable, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# dependencias NLP con tolerancia a fallos
# -----------------------------
spacy_ok = False
nlp = None
try:
    import spacy
    try:
        # intenta cargar modelo pequeño en español (si no está, cae a fallback)
        nlp = spacy.load("es_core_news_sm")
        spacy_ok = True
    except Exception:
        spacy_ok = False
except Exception:
    spacy_ok = False

nltk_ok = True
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem.snowball import SpanishStemmer
    # intenta asegurar stopwords (si no están bajadas, habrá excepción al usarlas)
    try:
        _ = nltk_stopwords.words("spanish")
    except LookupError:
        nltk.download("stopwords")
    stemmer = SpanishStemmer()
except Exception:
    nltk_ok = False
    stemmer = None

# stopwords mínimas de emergencia (si falla todo)
STOPWORDS_BACKUP = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no","una",
    "su","al","lo","como","más","pero","sus","le","ya","o","fue","este","ha","sí","porque","esta",
    "son","entre","cuando","muy","sin","sobre","también","me","hasta","hay","donde","quien","desde"
}

# -----------------------------
# utilidades de carpeta/plots
# -----------------------------

def asegurar_dir_en_script(nombre: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, nombre)
    os.makedirs(out, exist_ok=True)
    return out

def ruta_resultado(nombre: str, outdir: str) -> str:
    return os.path.join(outdir, nombre)

# -----------------------------
# datos
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
        "Varón 60 años, tos seca, disnea leve, saturación 93%, se indica radiografía de tórax y pruebas COVID.",
        "Mujer 25 años, cefalea intensa y fotofobia; se sugiere hidratación y control neurológico.",
        "Paciente 70 años con hipertensión y diabetes, control irregular de glicemia, fatiga y poliuria.",
        "Hombre 38 años, dolor torácico no irradiado, ECG sin cambios, se descarta síndrome coronario agudo.",
        "Niña 7 años, otalgia derecha, membrana timpánica eritematosa; se inicia amoxicilina.",
        "Mujer 55 años, dolor lumbar mecánico, sin déficit neurológico, se indica reposo y AINEs.",
        "Paciente 42 años, náuseas y vómitos postprandiales, prueba positiva para Helicobacter pylori.",
        "Hombre 29 años, esguince de tobillo tras actividad deportiva, edema y dolor a la palpación."
    ]

# -----------------------------
# limpieza + pipeline spaCy/NLTK
# -----------------------------

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.I)
URL_RE   = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
NUM_RE   = re.compile(r"\b\d+\b")
PUNCT_RE = re.compile(r"[^\w\sáéíóúüñ]", flags=re.UNICODE)

def limpieza_basica(texto: str) -> str:
    """Minúsculas + remoción de e-mails, URLs, números aislados y puntuación."""
    t = texto.lower()
    t = EMAIL_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = NUM_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def stopwords_es() -> set:
    """Obtiene stopwords de spaCy o NLTK; cae a un conjunto mínimo si ambos fallan."""
    sw = set()
    if spacy_ok and nlp is not None:
        try:
            sw |= {w for w in nlp.Defaults.stop_words}
        except Exception:
            pass
    if nltk_ok:
        try:
            sw |= set(nltk_stopwords.words("spanish"))
        except Exception:
            pass
    if not sw:
        sw = set(STOPWORDS_BACKUP)
    return sw

def tokenizar_lemmatizar(texto_limpio: str, sw: set) -> List[str]:
    """
    Tokeniza y lematiza con spaCy si es posible; de lo contrario usa un
    fallback sencillo con NLTK (stemming) o regex.
    """
    tokens: List[str] = []
    if spacy_ok and nlp is not None:
        doc = nlp(texto_limpio)
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            lemma = tok.lemma_.strip()
            if not lemma or lemma in sw:
                continue
            tokens.append(lemma)
        return tokens

    # Fallback sin spaCy
    piezas = texto_limpio.split()
    if nltk_ok and stemmer is not None:
        for p in piezas:
            if p and (p not in sw):
                tokens.append(stemmer.stem(p))
    else:
        tokens = [p for p in piezas if p and (p not in sw)]
    return tokens

# -----------------------------
# vectorización y reportes
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_tfidf(tokens_por_doc: List[List[str]], ngram_max: int = 2):
    """
    Vectoriza usando TF-IDF sobre tokens preprocesados.
    Se pasa 'analyzer' como función identidad para aceptar listas de tokens.
    """
    vec = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None, ngram_range=(1, ngram_max))
    tfidf = vec.fit_transform(tokens_por_doc)  # (n_docs, n_terms)
    features = vec.get_feature_names_out().tolist()
    return vec, tfidf, features

def graficar_top_terms_por_doc(tfidf, features, outdir: str, k: int = 10, indices: Optional[Iterable[int]] = None):
    """Crea barras de los k términos con mayor TF-IDF por documento."""
    n_docs = tfidf.shape[0]
    if indices is None:
        indices = range(n_docs)
    for i in indices:
        fila = tfidf[i, :].toarray().ravel()
        top_ids = np.argsort(fila)[-k:][::-1]
        terms = [features[j] for j in top_ids]
        vals = fila[top_ids]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(terms)), vals)
        plt.xticks(range(len(terms)), terms, rotation=45, ha="right")
        plt.ylabel("tf-idf")
        plt.title(f"doc {i}: top {k} términos TF-IDF")
        plt.tight_layout()
        plt.savefig(ruta_resultado(f"top_terminos_doc_{i}.png", outdir), dpi=150, bbox_inches="tight")
        plt.close()

# -----------------------------
# métricas comparativas corpus original vs preprocesado
# -----------------------------

def stats_tokens(lista_tokens: List[List[str]]) -> dict:
    tot_tokens = sum(len(t) for t in lista_tokens)
    vocab = Counter([tok for doc in lista_tokens for tok in doc])
    n_docs = len(lista_tokens)
    avg_len = tot_tokens / max(1, n_docs)
    vocab_size = len(vocab)
    ttr = vocab_size / max(1, tot_tokens)  # type-token ratio
    top_freq = vocab.most_common(15)
    return {
        "n_docs": n_docs,
        "tokens_totales": int(tot_tokens),
        "longitud_media_doc": float(avg_len),
        "vocabulario": int(vocab_size),
        "type_token_ratio": float(ttr),
        "top_15": top_freq,
    }

def tokenizar_simple(texto: str) -> List[str]:
    # tokenización muy básica para el "original" (previo a limpieza fuerte)
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", texto.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t.split()

# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="sesión 8.2 — spaCy/NLTK + TF-IDF en textos clínicos")
    parser.add_argument("--corpus_txt", type=str, default=None,
                        help="ruta a .txt con un documento por línea; si se omite, se usa un corpus de ejemplo")
    parser.add_argument("--ngram_max", type=int, default=2, help="tamaño máximo de n-grama para TF-IDF (1..N)")
    parser.add_argument("--k_top", type=int, default=10, help="cantidad de términos por documento a graficar")
    parser.add_argument("--graficar_todos", action="store_true",
                        help="si se pasa, grafica top términos para todos los documentos (no solo los 3 primeros)")
    args = parser.parse_args()

    outdir = asegurar_dir_en_script("resultados_sesion2")

    # 1) corpus
    if args.corpus_txt and os.path.exists(args.corpus_txt):
        docs_raw = cargar_corpus_desde_txt(args.corpus_txt)
    else:
        docs_raw = corpus_por_defecto()
    etiquetas = [f"doc{i}" for i in range(len(docs_raw))]

    # 2) limpieza básica
    docs_clean = [limpieza_basica(t) for t in docs_raw]

    # 3) stopwords y lematización/tokenización
    sw = stopwords_es()
    docs_tokens = [tokenizar_lemmatizar(t, sw) for t in docs_clean]

    # 4) TF-IDF con n-gramas
    vec, tfidf, features = vectorizar_tfidf(docs_tokens, ngram_max=args.ngram_max)

    # 5) visualización términos más relevantes
    if args.graficar_todos:
        indices = range(len(docs_raw))
    else:
        indices = range(min(3, len(docs_raw)))
    graficar_top_terms_por_doc(tfidf, features, outdir, k=args.k_top, indices=indices)

    # 6) comparación original vs preprocesado
    tokens_original = [tokenizar_simple(t) for t in docs_raw]
    stats_orig = stats_tokens(tokens_original)
    stats_proc = stats_tokens(docs_tokens)

    # 7) guardar matrices y resúmenes
    # dimensiones TF-IDF
    with open(ruta_resultado("tfidf_matrix_shape.txt", outdir), "w", encoding="utf-8") as f:
        f.write(f"{tfidf.shape[0]} {tfidf.shape[1]}\n")
    # vocabulario
    with open(ruta_resultado("vocabulario.json", outdir), "w", encoding="utf-8") as f:
        json.dump({"features": features}, f, ensure_ascii=False, indent=2)
    # corpus limpio/tokenizado
    with open(ruta_resultado("corpus_limpio.json", outdir), "w", encoding="utf-8") as f:
        json.dump({"docs_limpios": docs_clean, "tokens": docs_tokens}, f, ensure_ascii=False, indent=2)
    # resumen comparativo
    resumen = {
        "spacy_activado": bool(spacy_ok),
        "n_docs": len(docs_raw),
        "ngram_max": args.ngram_max,
        "k_top": args.k_top,
        "estadisticas": {
            "original": stats_orig,
            "preprocesado": stats_proc
        },
        "nota": "Se usó spaCy si estaba disponible; de lo contrario, fallback con NLTK/regex."
    }
    with open(ruta_resultado("resumen.json", outdir), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # 8) figura comparativa rápida (longitud media y TTR)
    plt.figure(figsize=(6.5, 4))
    x = np.arange(2)
    lon = [stats_orig["longitud_media_doc"], stats_proc["longitud_media_doc"]]
    ttr = [stats_orig["type_token_ratio"],  stats_proc["type_token_ratio"]]
    plt.bar(x - 0.18, lon, width=0.36, label="longitud media")
    plt.bar(x + 0.18, ttr, width=0.36, label="TTR")
    plt.xticks(x, ["original", "preprocesado"])
    plt.title("comparación corpus")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_resultado("comparacion_original_vs_preprocesado.png", outdir), dpi=150, bbox_inches="tight")
    plt.close()

    # 9) salida por consola
    print("\nresultados en:", outdir)
    print("- top_terminos_doc_*.png")
    print("- comparacion_original_vs_preprocesado.png")
    print("- tfidf_matrix_shape.txt")
    print("- vocabulario.json")
    print("- corpus_limpio.json")
    print("- resumen.json\n")

if __name__ == "__main__":
    main()