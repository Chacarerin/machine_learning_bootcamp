# actividad módulo 8 / sesión 3 — clasificación de reseñas clínicas con Transformers (Hugging Face)
#
# objetivos:
# - construir o cargar un conjunto breve (≥10) de reseñas clínicas en español
# - cargar el modelo "nlptown/bert-base-multilingual-uncased-sentiment" (1–5 estrellas) y ejecutar inferencia
# - presentar resultados legibles (texto + predicción + score) y guardar una tabla consolidada (CSV/JSON)
# - graficar la distribución de clases (1–5 estrellas) y una versión agregada (negativo/neutral/positivo)
# - comparar contra un “criterio humano” simple (heurística de palabras) para discutir acuerdos/limitaciones

import os
# Fuerza a Transformers a NO intentar TensorFlow/Keras (evita el error con Keras 3)
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import json
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# verificación de dependencias
# -----------------------------
def _check_deps():
    """
    Verifica que PyTorch y Transformers estén instalados. Transformers usará PyTorch como backend
    gracias a TRANSFORMERS_NO_TF=1 y 'framework="pt"' en pipeline.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        print("\nerror: falta PyTorch. instalar con:\n  pip install torch torchvision torchaudio\n")
        raise
    try:
        import transformers  # noqa: F401
    except Exception:
        print("\nerror: falta transformers. instalar con:\n  pip install transformers\n")
        raise

_check_deps()
from transformers import pipeline  # noqa: E402

# -----------------------------
# utilidades
# -----------------------------
def asegurar_dir_en_script(nombre: str) -> str:
    """Crea una carpeta dentro del mismo directorio del archivo actual."""
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, nombre)
    os.makedirs(out, exist_ok=True)
    return out

def ruta_resultado(nombre: str, outdir: str) -> str:
    """Devuelve la ruta absoluta de un artefacto dentro de la carpeta de resultados."""
    return os.path.join(outdir, nombre)

# -----------------------------
# datos
# -----------------------------
def reseñas_por_defecto() -> List[str]:
    """Corpus simulado de reseñas clínicas (≥10)."""
    return [
        "La atención fue excelente, resolvieron mis dudas y el tratamiento fue efectivo.",
        "Me sentí ignorado por el personal médico, no volvería al centro.",
        "El tiempo de espera fue razonable y el doctor explicó todo con claridad.",
        "Muy mala experiencia, extraviaron mis exámenes y nadie se hizo responsable.",
        "El equipo de enfermería fue muy amable, salí tranquila y bien informada.",
        "Atención correcta pero la sala de espera estaba muy llena y hacía calor.",
        "Tuvieron un trato impersonal y apresurado, me dejaron con más preguntas.",
        "Todo rápido y ordenado, agradezco la puntualidad y el seguimiento posterior.",
        "Los costos no fueron transparentes, me cobraron más de lo informado.",
        "Instalaciones limpias y modernas; el procedimiento resultó indoloro.",
        "El médico mostró empatía y se tomó el tiempo para explicar opciones.",
        "Falta de coordinación en admisión, tuve que repetir datos varias veces."
    ]

def cargar_reseñas_desde_txt(path: str) -> List[str]:
    """Carga un .txt con una reseña por línea (ignora líneas vacías)."""
    rese = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                rese.append(l)
    return rese

def cargar_reseñas_desde_csv(path: str, columna: str = "texto") -> List[str]:
    """Carga un .csv con la columna indicada que contiene los textos."""
    df = pd.read_csv(path)
    if columna not in df.columns:
        raise ValueError(f"la columna '{columna}' no existe en el csv")
    textos = [str(t).strip() for t in df[columna].tolist() if isinstance(t, str) and str(t).strip()]
    return textos

# -----------------------------
# inferencia con Transformers
# -----------------------------
def cargar_pipeline(model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Carga el pipeline de sentiment analysis (5 clases: 1..5 estrellas) y fuerza PyTorch.
    """
    clf = pipeline("sentiment-analysis", model=model_name, framework="pt")
    return clf

STAR2POL = {1: "negativo", 2: "negativo", 3: "neutral", 4: "positivo", 5: "positivo"}

def _label_to_star(label: str) -> int:
    """
    Convierte etiquetas del modelo (p. ej., '5 stars') a entero 1..5.
    """
    m = re.search(r"(\d)", label)
    if not m:
        return 3
    v = int(m.group(1))
    return min(5, max(1, v))

# -----------------------------
# criterio humano (heurística simple)
# -----------------------------
POS = {"excelente","efectivo","amable","tranquila","bien","rápido","ordenado","puntualidad","seguimiento","modernas","indoloro","claridad","limpias","empatía"}
NEG = {"mala","ignorado","extraviaron","responsable","llena","calor","impersonal","apresurado","cobraron","no","preguntas","costos","repetir","falta"}

def juicio_heuristico(texto: str) -> str:
    """
    Estima polaridad (negativo/neutral/positivo) mediante conteo de palabras clave.
    Es un proxy simple para comparar tendencias con el modelo.
    """
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", texto.lower())
    tokens = set(t.split())
    pos_hits = len(tokens & POS)
    neg_hits = len(tokens & NEG)
    if pos_hits > neg_hits:
        return "positivo"
    if neg_hits > pos_hits:
        return "negativo"
    return "neutral"

# -----------------------------
# visualizaciones
# -----------------------------
def graf_dispersion(df: pd.DataFrame, outpath: str):
    """Distribución de estrellas (1–5) predichas por el modelo."""
    conteo = df["stars"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    conteo.plot(kind="bar")
    plt.title("distribución de clases (1–5 estrellas)")
    plt.xlabel("estrellas"); plt.ylabel("frecuencia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def graf_polaridad(df: pd.DataFrame, outpath: str):
    """Distribución de polaridad agregada."""
    conteo = df["polarity"].value_counts().reindex(["negativo","neutral","positivo"]).fillna(0)
    plt.figure(figsize=(6, 4))
    conteo.plot(kind="bar")
    plt.title("distribución de polaridad")
    plt.xlabel("clase"); plt.ylabel("frecuencia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# ejecución principal
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="sesión 8.3 — clasificación de reseñas con Transformers (PyTorch)")
    parser.add_argument("--txt", type=str, default=None, help="ruta a .txt (una reseña por línea)")
    parser.add_argument("--csv", type=str, default=None, help="ruta a .csv con columna 'texto' (o indicar --columna)")
    parser.add_argument("--columna", type=str, default="texto", help="nombre de la columna si se usa --csv")
    parser.add_argument("--modelo", type=str, default="nlptown/bert-base-multilingual-uncased-sentiment",
                        help="modelo de Hugging Face a usar")
    args = parser.parse_args()

    outdir = asegurar_dir_en_script("resultados_sesion3")

    # 1) cargar reseñas
    if args.txt and os.path.exists(args.txt):
        textos = cargar_reseñas_desde_txt(args.txt)
    elif args.csv and os.path.exists(args.csv):
        textos = cargar_reseñas_desde_csv(args.csv, args.columna)
    else:
        textos = reseñas_por_defecto()

    if len(textos) < 10:
        print(f"[aviso] se recomiendan ≥10 reseñas; se encontraron {len(textos)}")

    # 2) cargar pipeline (PyTorch)
    print("cargando modelo y pipeline de sentiment (PyTorch)...")
    clf = cargar_pipeline(args.modelo)

    # 3) inferencia
    print("inferiendo sentimientos...")
    outputs = clf(textos, truncation=True)  # lista de dicts: {"label": "...", "score": float}

    # 4) tabla de resultados
    rows: List[Dict] = []
    for t, o in zip(textos, outputs):
        star = _label_to_star(o["label"])
        pol = STAR2POL.get(star, "neutral")
        human = juicio_heuristico(t)
        rows.append({
            "texto": t,
            "label_raw": o["label"],
            "score": float(o["score"]),
            "stars": int(star),
            "polarity": pol,
            "heuristico_polaridad": human,
            "acuerdo_humano": (human == pol)
        })
    df = pd.DataFrame(rows)

    # 5) guardar tabla y resumen
    df.to_csv(ruta_resultado("predicciones.csv", outdir), index=False, encoding="utf-8")

    # 6) gráficos
    graf_dispersion(df, ruta_resultado("distribucion_estrellas.png", outdir))
    graf_polaridad(df, ruta_resultado("distribucion_polaridad.png", outdir))

    # 7) resumen json con métricas simples de “acuerdo” con el criterio heurístico
    acuerdo = float(df["acuerdo_humano"].mean()) if len(df) else 0.0
    resumen = {
        "modelo": args.modelo,
        "n_reseñas": int(len(df)),
        "recuento_estrellas": {int(k): int(v) for k, v in df["stars"].value_counts().sort_index().to_dict().items()},
        "recuento_polaridad": {k: int(v) for k, v in df["polarity"].value_counts().to_dict().items()},
        "acuerdo_heuristico": acuerdo,
        "nota": (
            "El acuerdo con la heurística es un proxy simple del 'criterio humano'. "
            "Sirve para reflexionar sobre sesgos/limitaciones; no es evaluación definitiva."
        )
    }
    with open(ruta_resultado("resumen.json", outdir), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # 8) muestra por consola (top-5 por score)
    print("\n== ejemplos (ordenados por score) ==")
    for _, r in df.sort_values("score", ascending=False).head(5).iterrows():
        print(f"[{r['stars']}★/{r['polarity']}] score={r['score']:.3f} :: {r['texto'][:120]}")

    # 9) artefactos generados
    print("\nlisto. resultados en:", outdir)
    print("- predicciones.csv")
    print("- distribucion_estrellas.png")
    print("- distribucion_polaridad.png")
    print("- resumen.json\n")

if __name__ == "__main__":
    main()