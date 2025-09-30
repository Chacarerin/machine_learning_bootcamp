# actividad modulo 8 - clasificacion de notas clinicas con enfoque etico y sesgos

# objetivo general:
# - construir un pipeline de clasificacion de textos clinicos para predecir la gravedad ('leve', 'moderado', 'severo')
# - evaluar el desempeño global y por subgrupos (genero y grupos etarios) para detectar posibles sesgos
# - generar artefactos reproducibles (graficos, csv, json, npy) para incluir en el readme y la entrega
# - usar al menos dos enfoques de representacion y modelado (p.ej., tf-idf + naive bayes, embeddings + svm, bert, etc.)

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# lime opcional
# nota: si lime no esta instalado, el codigo continua sin explicaciones locales
try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

# nltk opcional para stopwords y stemming
# nota: si falla la descarga de stopwords o el modulo, se usa un conjunto vacio y se omite stemming
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SpanishStemmer
    _stopwords_es = set(stopwords.words('spanish'))
except Exception:
    _stopwords_es = set()
    SpanishStemmer = None

# embeddings word2vec opcionales (entrenados sobre el propio corpus)
try:
    from gensim.models import Word2Vec
except Exception:
    Word2Vec = None

# transformers opcional (bert en espanol)
# nota: si no hay gpu o no esta instalado transformers/torch, se desactiva y se sigue con el modelo base
_transformers_ok = True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    _transformers_ok = True
except Exception:
    _transformers_ok = False

# paso 0) semilla global para reproducibilidad
SEED = 42
random.seed(SEED); np.random.seed(SEED)


def asegurar_directorio_en_script(nombre: str) -> Path:
    # crea una carpeta de salida dentro del mismo directorio del archivo actual (no depende del cwd)
    # entrada: nombre de la subcarpeta a crear
    # salida: objeto Path con la ruta creada (existe o es creada)
    base = Path(os.path.dirname(__file__))
    out = base / nombre
    out.mkdir(parents=True, exist_ok=True)
    return out


def cargar_dataset(nombre_csv: str) -> pd.DataFrame:
    # carga el csv con el dataset clinico y valida columnas requeridas
    # - si la ruta es relativa, se resuelve contra la carpeta del script
    # - debe contener: texto_clinico, edad, genero, afeccion, gravedad
    if not os.path.isabs(nombre_csv):
        nombre_csv = os.path.join(os.path.dirname(__file__), nombre_csv)
    df = pd.read_csv(nombre_csv, encoding='utf-8')
    df.columns = [c.strip().lower() for c in df.columns]
    cols_req = {'texto_clinico', 'edad', 'genero', 'afeccion', 'gravedad'}
    faltantes = cols_req.difference(df.columns)
    if faltantes:
        raise ValueError(f'falta(n) columna(s): {faltantes}')
    return df


def binarizar_edad(df: pd.DataFrame) -> pd.Series:
    # construye grupos etarios discretos para auditoria de sesgos
    # cortes: <=29, 30-44, 45-64, 65+
    bins = [0, 29, 44, 64, 120]
    labels = ['<=29', '30-44', '45-64', '65+']
    return pd.cut(df['edad'].astype(int), bins=bins, labels=labels, include_lowest=True)


def preprocesar_texto_basico(texto: str) -> str:
    # preprocesamiento simple y robusto:
    # - pasa a minusculas
    # - tokeniza con split basico
    # - filtra tokens con longitud > 2 y que no sean stopwords (si estan disponibles)
    # - aplica stemming opcional en espanol si hay nltk
    t = ''.join(ch.lower() for ch in str(texto))
    tokens = [tok for tok in t.split() if len(tok) > 2 and (tok not in _stopwords_es)]
    if SpanishStemmer is not None:
        stemmer = SpanishStemmer()
        tokens = [stemmer.stem(tok) for tok in tokens]
    return ' '.join(tokens)


def entrenar_nb_tfidf(textos_train, y_train):
    # modelo base: pipeline tf-idf + naive bayes multinomial
    # - vectorizador: unigrams y bigrams, min_df=2 para reducir ruido
    # - se pasan stopwords y preprocesador definido arriba
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words=list(_stopwords_es) if _stopwords_es else None,
            preprocessor=preprocesar_texto_basico,
            ngram_range=(1, 2),
            min_df=2
        )),
        ('clf', MultinomialNB())
    ])
    pipe.fit(textos_train, y_train)
    return pipe


def construir_word2vec(textos: list, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 30):
    # entrena word2vec sobre el propio corpus para dejar evidencia de embeddings
    # si gensim no esta disponible, devuelve None y el flujo sigue
    if Word2Vec is None:
        return None
    tokenizados = [preprocesar_texto_basico(t).split() for t in textos]
    model = Word2Vec(sentences=tokenizados, vector_size=vector_size, window=window, min_count=min_count, workers=1, seed=SEED)
    model.train(tokenizados, total_examples=len(tokenizados), epochs=epochs)
    return model


def vector_promedio_w2v(texto: str, modelo_w2v) -> np.ndarray:
    # representa un texto como el promedio de los vectores de sus tokens presentes en el vocabulario
    # si ningun token esta en el modelo, retorna un vector de ceros de la dimension correspondiente
    vecs = []
    for tok in preprocesar_texto_basico(texto).split():
        if tok in modelo_w2v.wv:
            vecs.append(modelo_w2v.wv[tok])
    if not vecs:
        return np.zeros(modelo_w2v.vector_size, dtype=float)
    return np.mean(vecs, axis=0)


def guardar_matriz_confusion(y_true, y_pred, etiquetas, titulo, outpath: Path):
    # genera y guarda una matriz de confusion como imagen
    # - usa 'viridis' para contraste
    # - agrega anotaciones numericas por celda
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='viridis')
    ax.set_title(titulo)
    ax.set_xlabel('prediccion'); ax.set_ylabel('real')
    ax.set_xticks(range(len(etiquetas))); ax.set_xticklabels(etiquetas)
    ax.set_yticks(range(len(etiquetas))); ax.set_yticklabels(etiquetas)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def evaluar_sesgo(df_eval: pd.DataFrame, grupo_col: str, etiquetas):
    # calcula metricas macro por subgrupo (precision, recall, f1) para detectar posibles brechas de desempeño
    # entrada: df con columnas y_true e y_pred, ademas de la columna de grupo (p.ej., genero o grupo_edad)
    # salida: dataframe ordenado por f1 macro descendente, con soporte por subgrupo
    filas = []
    for g, df_g in df_eval.groupby(grupo_col):
        rep = classification_report(df_g['y_true'], df_g['y_pred'], labels=etiquetas, output_dict=True, zero_division=0)
        filas.append({
            'grupo': f'{grupo_col}:{g}',
            'precision_macro': rep['macro avg']['precision'],
            'recall_macro': rep['macro avg']['recall'],
            'f1_macro': rep['macro avg']['f1-score'],
            'soporte': len(df_g)
        })
    return pd.DataFrame(filas).sort_values('f1_macro', ascending=False)


# dataset para transformers (si disponible)
class SimpleTextDataset(torch.utils.data.Dataset if _transformers_ok else object):
    # dataset simple para trainer de transformers
    # guarda textos y labels ya mapeados a ids, aplica tokenizacion en __getitem__
    def __init__(self, textos=None, labels=None, tokenizer=None, max_len=160):
        if not _transformers_ok:
            return
        self.textos = textos; self.labels = labels
        self.tokenizer = tokenizer; self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        t = str(self.textos[idx])
        enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def entrenar_transformer(textos_train, y_train, textos_val, y_val, etiquetas, outdir: Path):
    # entrena un modelo bert en espanol (dccuchile/bert-base-spanish-wwm-cased) si transformers esta disponible
    # hiperparametros ajustados para ejecucion rapida (2 epocas, batch 8)
    # retorna (tokenizer, model) o (None, None) si falla
    if not _transformers_ok:
        return None, None
    try:
        modelo_id = 'dccuchile/bert-base-spanish-wwm-cased'
        tokenizer = AutoTokenizer.from_pretrained(modelo_id)
        model = AutoModelForSequenceClassification.from_pretrained(modelo_id, num_labels=len(etiquetas))

        label2id = {etq: i for i, etq in enumerate(etiquetas)}
        y_train_ids = [label2id[y] for y in y_train]
        y_val_ids = [label2id[y] for y in y_val]

        train_ds = SimpleTextDataset(textos_train, y_train_ids, tokenizer, 160)
        val_ds = SimpleTextDataset(textos_val, y_val_ids, tokenizer, 160)

        args = TrainingArguments(
            output_dir=str(outdir / 'hf_runs'),
            evaluation_strategy='epoch',
            save_strategy='no',
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_steps=50,
            seed=SEED
        )

        def compute_metrics(eval_pred):
            # convierte logits a clases y calcula metricas macro (precision, recall, f1)
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            rep = classification_report(labels, preds, output_dict=True, zero_division=0)
            return {
                'precision_macro': rep['macro avg']['precision'],
                'recall_macro': rep['macro avg']['recall'],
                'f1_macro': rep['macro avg']['f1-score']
            }

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
        trainer.train()
        return tokenizer, model
    except Exception as e:
        print(f'aviso: no se pudo entrenar transformer: {e}')
        return None, None


def predecir_transformer(tokenizer, model, textos, etiquetas):
    # infiere etiquetas con el modelo transformer si esta disponible
    # mapea ids a etiquetas originales y retorna lista de strings
    if (tokenizer is None) or (model is None):
        return None
    try:
        preds = []
        for t in textos:
            enc = tokenizer(str(t), truncation=True, padding='max_length', max_length=160, return_tensors='pt')
            with torch.no_grad():
                logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
            pred_id = int(np.argmax(logits.cpu().numpy(), axis=1)[0])
            preds.append(pred_id)
        id2label = {i: etq for i, etq in enumerate(etiquetas)}
        return [id2label[i] for i in preds]
    except Exception as e:
        print(f'aviso: no se pudo predecir con transformer: {e}')
        return None


def explicar_con_lime(clf_pipeline, textos, etiquetas, outdir: Path, n=3):
    # genera explicaciones locales de lime para n textos aleatorios del conjunto de prueba
    # se guarda una lista de pesos por palabra en archivos .txt (compatibles con la entrega)
    if LimeTextExplainer is None:
        return
    explainer = LimeTextExplainer(class_names=etiquetas)  # sin language para compatibilidad
    n = min(n, len(textos))
    idxs = np.random.choice(len(textos), size=n, replace=False)
    for i in idxs:
        exp = explainer.explain_instance(textos[i], clf_pipeline.predict_proba, num_features=10)
        (outdir / f'lime_ejemplo_{i}.txt').write_text(str(exp.as_list()), encoding='utf-8')


def main():
    # orquesta todo el flujo y deja los artefactos en 'resultados_mod8' junto al script
    warnings.filterwarnings('ignore')
    outdir = asegurar_directorio_en_script('resultados_mod8')

    # 1) carga de datos
    # - si existe la variable de entorno DATASET_CLINICO, se usa esa ruta
    # - por defecto, se intenta 'dataset_clinico_simulado_200.csv' al lado del script
    ruta_csv = os.environ.get('DATASET_CLINICO', 'dataset_clinico_simulado_200.csv')
    df = cargar_dataset(ruta_csv)

    # 2) columnas y target
    # - texto_clinico como entrada
    # - gravedad como clase (tres niveles definidos en 'etiquetas' para ordenar reportes y graficos)
    texto_col = 'texto_clinico'
    target_col = 'gravedad'
    etiquetas = ['leve', 'moderado', 'severo']

    # 3) eda basico: distribucion de clases
    # - guarda un grafico .png y un json con los conteos por clase
    dist = df[target_col].value_counts().reindex(etiquetas, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    dist.plot(kind='bar', ax=ax)
    ax.set_title('distribucion de clases')
    ax.set_xlabel('clase'); ax.set_ylabel('frecuencia')
    fig.tight_layout(); fig.savefig(outdir / 'clases_distribucion.png', dpi=150); plt.close(fig)
    (outdir / 'clases_distribucion.json').write_text(json.dumps(dist.to_dict(), indent=2), encoding='utf-8')

    # 4) metadatos para sesgos
    # - crea grupo_edad con cortes discretos
    # - normaliza genero como texto
    df['grupo_edad'] = binarizar_edad(df)
    df['genero'] = df['genero'].astype(str)

    # 5) split
    # - separa train/test con estratificacion por la clase objetivo para mantener proporciones
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df[target_col])
    X_train = df_train[texto_col].values
    y_train = df_train[target_col].values
    X_test = df_test[texto_col].values
    y_test = df_test[target_col].values

    # 6) modelo 1: nb + tfidf
    # - entrena pipeline y guarda reporte (json) + matriz de confusion (png)
    nb_pipe = entrenar_nb_tfidf(X_train, y_train)
    y_pred_nb = nb_pipe.predict(X_test)
    guardar_matriz_confusion(y_test, y_pred_nb, etiquetas, 'matriz de confusion - nb (tfidf)', outdir / 'matriz_confusion_nb.png')
    rep_nb = classification_report(y_test, y_pred_nb, labels=etiquetas, output_dict=True, zero_division=0)
    (outdir / 'reporte_nb.json').write_text(json.dumps(rep_nb, indent=2), encoding='utf-8')

    # 7) embeddings word2vec (para cumplir requisito)
    # - entrena sobre textos de entrenamiento y exporta vectors promedio de X_test como .npy
    if Word2Vec is not None:
        w2v = construir_word2vec(df_train[texto_col].tolist(), vector_size=100)
        X_test_w2v = np.vstack([vector_promedio_w2v(t, w2v) for t in X_test])
        np.save(outdir / 'ejemplo_embeddings_w2v.npy', X_test_w2v)

    # 8) modelo 2: transformer (si disponible)
    # - entrena un bert en espanol con trainer de transformers
    # - guarda matriz de confusion y reporte si se pudo entrenar
    tokenizer, model = (None, None)
    if _transformers_ok:
        tokenizer, model = entrenar_transformer(X_train, y_train, X_test, y_test, etiquetas, outdir)
    y_pred_tr = predecir_transformer(tokenizer, model, X_test, etiquetas) if tokenizer else None
    if y_pred_tr is not None:
        guardar_matriz_confusion(y_test, y_pred_tr, etiquetas, 'matriz de confusion - transformer', outdir / 'matriz_confusion_transformer.png')
        rep_tr = classification_report(y_test, y_pred_tr, labels=etiquetas, output_dict=True, zero_division=0)
        (outdir / 'reporte_transformer.json').write_text(json.dumps(rep_tr, indent=2), encoding='utf-8')

    # 9) evaluacion de sesgos (mejor disponible)
    # - usa el mejor modelo disponible (transformer si existe, de lo contrario nb)
    # - calcula metricas macro por genero y grupo_edad y exporta a csv
    y_pred_final = y_pred_tr if y_pred_tr is not None else y_pred_nb
    df_eval = df_test[[texto_col, target_col, 'genero', 'grupo_edad']].copy()
    df_eval['y_true'] = y_test
    df_eval['y_pred'] = y_pred_final
    sesgo_genero = evaluar_sesgo(df_eval, 'genero', etiquetas)
    sesgo_edad = evaluar_sesgo(df_eval, 'grupo_edad', etiquetas)
    sesgo_genero.to_csv(outdir / 'sesgo_genero.csv', index=False)
    sesgo_edad.to_csv(outdir / 'sesgo_grupo_edad.csv', index=False)

    # 10) explicabilidad con lime
    # - genera hasta n=3 explicaciones locales en .txt si lime esta disponible
    explicar_con_lime(nb_pipe, list(X_test), etiquetas, outdir, n=3)

    # 11) resumen para readme
    # - crea un csv con las metricas macro de cada modelo
    filas = [{
        'modelo': 'naive_bayes_tfidf',
        'precision_macro': rep_nb['macro avg']['precision'],
        'recall_macro': rep_nb['macro avg']['recall'],
        'f1_macro': rep_nb['macro avg']['f1-score']
    }]
    if y_pred_tr is not None:
        rep_tr = classification_report(y_test, y_pred_tr, labels=etiquetas, output_dict=True, zero_division=0)
        filas.append({
            'modelo': 'transformer_es',
            'precision_macro': rep_tr['macro avg']['precision'],
            'recall_macro': rep_tr['macro avg']['recall'],
            'f1_macro': rep_tr['macro avg']['f1-score']
        })
    pd.DataFrame(filas).to_csv(outdir / 'metricas_resumen.csv', index=False)

    print('listo: resultados en', outdir.resolve())


if __name__ == '__main__':
    main()