# modulo 7 - sistema inteligente de scoring crediticio con redes neuronales profundas

# objetivo: cargar datos de credito, preprocesar, entrenar dnn y resnet tabular, evaluar y explicar con shap (y lime si esta disponible)


import os
import random
import json
import warnings
warnings.filterwarnings("ignore")

# se fija semilla para reproducibilidad
SEED = 42
random.seed(SEED)
import numpy as np
np.random.seed(SEED)

# librerias basicas
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# sklearn y herramientas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# opciones de balanceo (smote) si esta disponible
def try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE
    except Exception:
        return None

# keras / tensorflow
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras import layers, models, callbacks, regularizers

# shap y lime (se usan si estan disponibles)
def try_import_shap():
    try:
        import shap
        return shap
    except Exception:
        return None

def try_import_lime():
    try:
        from lime.lime_tabular import LimeTabularExplainer
        return LimeTabularExplainer
    except Exception:
        return None

# ------------------------------------------------------------------------------------
# util: crear carpeta de resultados junto al archivo actual (independiente del cwd)
# ------------------------------------------------------------------------------------
def asegurar_directorio_en_script(nombre: str) -> Path:
    # crea una carpeta dentro del mismo directorio del archivo actual
    base = Path(os.path.dirname(__file__))
    out = base / nombre
    out.mkdir(parents=True, exist_ok=True)
    return out

# ------------------------------------------------------------------------------------
# 1) carga de datos
# se prioriza openml (credit-g). si no hay internet, se intenta leer un csv local si existe.
# ------------------------------------------------------------------------------------
def cargar_datos():
    # intento 1: openml (credit-g)
    try:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("credit-g", version=1, as_frame=True).frame
        # openml entrega target en 'class' ('good'/'bad')
        return df, "openml:credit-g"
    except Exception:
        pass
    # intento 2: csv local (se busca en cwd)
    posibles = [
        "german_credit_data.csv",
        "german_credit_data.csv",
        "credit-g.csv"
    ]
    for p in posibles:
        if Path(p).exists():
            df = pd.read_csv(p)
            return df, f"local:{p}"
    raise RuntimeError("no se pudo cargar el dataset. intente con conexion a internet o coloque el csv en la carpeta del proyecto.")

# ------------------------------------------------------------------------------------
# 2) preprocesamiento
# - codificacion one-hot para categoricas
# - escalamiento para numericas
# - manejo de desbalance con class_weight (y opcional smote)
# ------------------------------------------------------------------------------------
def preparar_datos(df: pd.DataFrame):
    # renombrar columna objetivo estandar como 'target'
    if "class" in df.columns:
        y = df["class"].copy()
        X = df.drop(columns=["class"]).copy()
        # convertir a 1/0 (1 = buen pagador, 0 = mal pagador) para consistencia
        y = (y.astype(str).str.lower() == "good").astype(int)
    elif "target" in df.columns:
        y = df["target"].copy().astype(int)
        X = df.drop(columns=["target"]).copy()
    else:
        # si el conjunto trae otra etiqueta, se intenta detectarla
        target_candidates = [c for c in df.columns if c.lower() in ("label","y","default","bad","good")]
        if not target_candidates:
            raise ValueError("no se encontro columna objetivo ('class' o 'target').")
        t = target_candidates[0]
        y = df[t].copy()
        X = df.drop(columns=[t]).copy()
        # se intenta mapear a binario si es texto
        if y.dtype == "object":
            vals = y.astype(str).str.lower().unique().tolist()
            if "good" in vals and "bad" in vals:
                y = (y.astype(str).str.lower() == "good").astype(int)

    # separacion de tipos
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number","bool"]).columns.tolist()

    # pipelines
    num_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # se ajusta el preprocesador y se transforma
    X_train_proc = pre.fit_transform(X_train)
    X_test_proc = pre.transform(X_test)

    # nombres de columnas despues de one-hot (para interpretabilidad)
    ohe = pre.named_transformers_["cat"]["onehot"] if cat_cols else None
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist() if ohe is not None else []
    feature_names = num_cols + cat_feature_names

    # pesos por clase (para desbalanceo)
    classes = np.array(sorted(y_train.unique()))
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.values)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # aplicar smote opcional (solo si esta instalado)
    SMOTE = try_import_smote()
    if SMOTE is not None:
        try:
            sm = SMOTE(random_state=SEED)
            X_train_proc, y_train = sm.fit_resample(X_train_proc, y_train)
        except Exception:
            # si falla smote (por memoria u otra razon), se sigue con class_weight
            pass

    info = {
        "n_train": int(X_train_proc.shape[0]),
               "n_test": int(X_test_proc.shape[0]),
        "n_features": int(X_train_proc.shape[1]),
        "feature_names": feature_names,
        "class_weight": class_weight_dict,
        "cat_cols": cat_cols,
        "num_cols": num_cols
    }
    return (X_train_proc, X_test_proc, y_train.values.astype(int), y_test.values.astype(int), info, pre)

# ------------------------------------------------------------------------------------
# 3) modelos: dnn y resnet tabular
# ------------------------------------------------------------------------------------
def construir_dnn(input_dim: int):
    # dnn simple con regularizacion l2 y dropout
    reg = 1e-4
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(reg)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def residual_block(x, units, reg):
    # bloque residual basico para tabulares
    h = layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(reg))(x)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(units, activation=None, kernel_regularizer=regularizers.l2(reg))(h)
    # proyeccion si cambia la dimension
    if x.shape[-1] != units:
        x = layers.Dense(units, activation=None, kernel_regularizer=regularizers.l2(reg))(x)
    out = layers.Add()([x, h])
    out = layers.Activation("relu")(out)
    return out

def construir_resnet_tabular(input_dim: int):
    reg = 1e-4
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(reg))(inp)
    x = residual_block(x, 128, reg)
    x = residual_block(x, 128, reg)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ------------------------------------------------------------------------------------
# 4) entrenamiento, evaluacion y graficos
# ------------------------------------------------------------------------------------
def entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test, class_weight, nombre_modelo, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
    rlrop = callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5, monitor="val_loss")

    hist = modelo.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        verbose=0,
        class_weight=class_weight,
        callbacks=[es, rlrop]
    )

    # predicciones y metricas
    proba = modelo.predict(X_test, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)

    metrics = {
        "modelo": nombre_modelo,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, proba))
    }

    # matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"matriz de confusion - {nombre_modelo}")
    ax.set_xlabel("prediccion")
    ax.set_ylabel("real")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outdir / f"matriz_confusion_{nombre_modelo}.png", dpi=150)
    plt.close(fig)

    # curva roc
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title(f"curva roc - {nombre_modelo}")
    fig.tight_layout()
    fig.savefig(outdir / f"roc_{nombre_modelo}.png", dpi=150)
    plt.close(fig)

    # guardar historico
    (outdir / f"hist_{nombre_modelo}.json").write_text(json.dumps(hist.history, indent=2))

    return metrics, proba

# ------------------------------------------------------------------------------------
# 5) explicabilidad: shap (kernel explainer por compatibilidad general). lime opcional.
# ------------------------------------------------------------------------------------
def explicar_shap(modelo, X_train, X_test, feature_names, outdir: Path, n_background=200, n_samples=500):
    shap = try_import_shap()
    if shap is None:
        return {"shap_disponible": False}
    # para rapidez se muestrea un subconjunto
    bg_idx = np.random.choice(X_train.shape[0], size=min(n_background, X_train.shape[0]), replace=False)
    background = X_train[bg_idx]
    test_idx = np.random.choice(X_test.shape[0], size=min(n_samples, X_test.shape[0]), replace=False)
    X_test_sample = X_test[test_idx]

    f = lambda data: modelo.predict(data, verbose=0).ravel()
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_test_sample, nsamples=200)

    # grafico summary
    try:
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary_plot.png", dpi=150)
        plt.close()
    except Exception:
        pass

    return {"shap_disponible": True}

def explicar_lime(X_train, y_train, X_test, feature_names, outdir: Path):
    LimeTabularExplainer = try_import_lime()
    if LimeTabularExplainer is None:
        return {"lime_disponible": False}
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["bad","good"], discretize_continuous=True
    )
    # se crea un ejemplo explicativo sin dependencias html (se guarda texto)
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(X_test[i], lambda x: np.c_[1 - x.mean(axis=1), x.mean(axis=1)], num_features=10)
    txt = "\n".join([f"{feat}: {weight:.4f}" for feat, weight in exp.as_list()])
    (outdir / "lime_example.txt").write_text(txt)
    return {"lime_disponible": True}

# ------------------------------------------------------------------------------------
# 6) script principal
# ------------------------------------------------------------------------------------
def main():
    # crea carpeta de resultados en el mismo directorio del archivo actual
    outdir = asegurar_directorio_en_script("resultados_mod7")

    df, origen = cargar_datos()
    X_train, X_test, y_train, y_test, info, pre = preparar_datos(df)

    # modelos
    dnn = construir_dnn(info["n_features"])
    resnet = construir_resnet_tabular(info["n_features"])

    # entrenamiento + evaluacion
    met_dnn, proba_dnn = entrenar_y_evaluar(dnn, X_train, y_train, X_test, y_test, info["class_weight"], "dnn", outdir)
    met_res, proba_res = entrenar_y_evaluar(resnet, X_train, y_train, X_test, y_test, info["class_weight"], "resnet", outdir)

    # explicabilidad (shap kernel)
    explicar_shap(dnn, X_train, X_test, info["feature_names"], outdir)
    # lime opcional
    explicar_lime(X_train, y_train, X_test, info["feature_names"], outdir)

    # guardar metricas
    dfm = pd.DataFrame([met_dnn, met_res])
    dfm.to_csv(outdir / "metricas_modelos.csv", index=False)

    # resumen en consola
    print("---- resumen ejecucion ----")
    print(f"origen datos: {origen}")
    print(f"registros train/test: {info['n_train']}/{info['n_test']}")
    print(dfm.to_string(index=False))
    print("archivos generados en:", str(outdir.resolve()))
    print(" - matriz_confusion_dnn.png, roc_dnn.png, shap_summary_plot.png")
    print(" - matriz_confusion_resnet.png, roc_resnet.png")
    print(" - hist_dnn.json, hist_resnet.json, metricas_modelos.csv")
    print(" - lime_example.txt (si lime esta disponible)")

if __name__ == '__main__':
    main()