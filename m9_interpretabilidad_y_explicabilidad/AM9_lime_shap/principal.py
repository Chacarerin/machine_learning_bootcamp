# modulo 9 - interpretabilidad de modelos predictivos con lime y shap
# objetivo: entrenar un clasificador sobre heart failure (kaggle), explicar con shap y lime (3 casos), y evaluar sesgo
# estilo: un solo archivo, comentarios en minusculas, pasos claros, resultados al lado del script

import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# reproducibilidad
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# intentos opcionales (xgboost, shap, lime) -> si no estan instalados, el flujo sigue con alternativas
def try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        return None

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
# util: crear carpeta de resultados junto al archivo actual
# ------------------------------------------------------------------------------------
def asegurar_directorio_en_script(nombre: str) -> Path:
    # crea una carpeta dentro del mismo directorio del archivo actual
    base = Path(os.path.dirname(__file__))
    out = base / nombre
    out.mkdir(parents=True, exist_ok=True)
    return out

# ------------------------------------------------------------------------------------
# utils dataset heart (busqueda local + fallback a kagglehub)
# ------------------------------------------------------------------------------------
def _buscar_archivo_heart():
    # permite forzar ruta con variable de entorno
    candidatos = [
        os.environ.get("DATASET_HEART"),
        "heart.csv",
        "heart_failure_clinical_records_dataset.csv",
        "heart-failure-prediction.csv",
        "heart_failure_prediction.csv",
    ]
    # 1) cwd
    for c in candidatos:
        if c:
            p = Path(c)
            if p.exists():
                return p.resolve()
    # 2) mismo directorio del script
    base = Path(os.path.dirname(__file__))
    for c in candidatos:
        if c:
            p = base / c
            if p.exists():
                return p.resolve()
    return None

def cargar_heart():
    # intenta local; si no existe, descarga con kagglehub
    ruta = _buscar_archivo_heart()
    if ruta is None:
        try:
            import kagglehub
            print("descargando dataset desde kaggle (fedesoriano/heart-failure-prediction)...")
            path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
            # nombre tipico en ese dataset
            candidatos = [
                "heart.csv",
                "heart_failure_clinical_records_dataset.csv",
                "heart-failure-prediction.csv",
                "heart_failure_prediction.csv",
            ]
            for c in candidatos:
                p = Path(path) / c
                if p.exists():
                    ruta = p
                    break
            if ruta is None or not Path(ruta).exists():
                raise RuntimeError("no se encontro el csv dentro de la carpeta descargada")
        except Exception as e:
            raise RuntimeError(f"no se encontro dataset local ni se pudo descargar de kaggle: {e}")

    df = pd.read_csv(ruta)
    df.columns = [c.strip().lower() for c in df.columns]

    # target tipico en kaggle: 'heartdisease'
    target_candidates = [c for c in df.columns if c in ("heartdisease", "target", "label", "y")]
    if not target_candidates:
        raise ValueError("no se encontro la columna objetivo (ej: 'heartdisease').")
    target_col = target_candidates[0]

    # limpieza minima
    df = df.dropna(axis=0).copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)

    return df, target_col

# ------------------------------------------------------------------------------------
# 2) preprocesamiento y split
# - one-hot para categoricas
# - escalado para numericas
# - train/test estratificado
# - se devuelve X,y, preprocesador y nombres de features post one-hot
# ------------------------------------------------------------------------------------
def preparar_datos(df: pd.DataFrame, target_col: str):
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # binarizacion si la etiqueta viene como texto
    if y.dtype == "object":
        yl = y.astype(str).str.lower()
        if set(yl.unique()) <= {"1","0","yes","no","true","false","si","no"}:
            y = yl.map({"1":1,"0":0,"yes":1,"no":0,"true":1,"false":0,"si":1}).astype(int)
        else:
            y = pd.to_numeric(y, errors="raise").astype(int)

    # separacion de tipos para el preprocesamiento
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number","bool"]).columns.tolist()

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    X_train_proc = pre.fit_transform(X_train)
    X_test_proc  = pre.transform(X_test)

    # nombres de columnas despues de one-hot
    if cat_cols:
        ohe = pre.named_transformers_["cat"]["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        cat_names = []
    feature_names = num_cols + cat_names

    info = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_names": feature_names
    }
    return (X_train, X_test, y_train.values.astype(int), y_test.values.astype(int),
            X_train_proc, X_test_proc, pre, info)

# ------------------------------------------------------------------------------------
# 3) construccion de modelo
# - por defecto random forest (arboles permiten shap tree explainer eficiente)
# - alternativa: logistic regression
# - si esta xgboost, se usa como opcion avanzada
# ------------------------------------------------------------------------------------
def construir_modelo(input_dim: int, preferencia: str = "rf"):
    XGBClassifier = try_import_xgb()
    if preferencia == "xgb" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            random_state=SEED, eval_metric="logloss"
        )
    if preferencia == "lr":
        return LogisticRegression(max_iter=500, random_state=SEED, n_jobs=None)
    # default rf
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    )

# ------------------------------------------------------------------------------------
# 4) entrenamiento y evaluacion del modelo
# - calcula accuracy, precision, recall, f1, auc
# - guarda matriz de confusion y curva roc
# ------------------------------------------------------------------------------------
def entrenar_y_evaluar(modelo, X_train_proc, y_train, X_test_proc, y_test, outdir: Path, nombre: str):
    modelo.fit(X_train_proc, y_train)
    proba = modelo.predict_proba(X_test_proc)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test_proc)
    y_pred = (proba >= 0.5).astype(int)

    metrics = {
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, proba))
    }

    # matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(f"matriz de confusion - {nombre}")
    ax.set_xlabel("prediccion"); ax.set_ylabel("real")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / f"matriz_confusion_{nombre}.png", dpi=150)
    plt.close(fig)

    # curva roc
    fig, ax = plt.subplots(figsize=(5,4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title(f"curva roc - {nombre}")
    fig.tight_layout()
    fig.savefig(outdir / f"roc_{nombre}.png", dpi=150)
    plt.close(fig)

    (outdir / f"metricas_{nombre}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, y_pred, proba

# ------------------------------------------------------------------------------------
# 5) shap: explicaciones globales y locales
# - global: summary (dot) + bar
# - local: 3 casos (waterfall si es posible)
# ------------------------------------------------------------------------------------
def explicar_shap(modelo, X_train_proc, X_test_proc, feature_names, outdir: Path, n_local=3):
    shap = try_import_shap()
    if shap is None:
        return {"shap_disponible": False}

    try:
        # para modelos de arbol: tree explainer es mas eficiente
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_test_proc)
        # compatibilidad: para binario, shap_values puede ser lista [neg, pos] o matriz
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # summary dot
        shap.summary_plot(sv, X_test_proc, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary_dot.png", dpi=150)
        plt.close()

        # summary bar
        shap.summary_plot(sv, X_test_proc, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary_bar.png", dpi=150)
        plt.close()

        # locales: waterfall para 3 observaciones aleatorias
        idxs = np.random.choice(X_test_proc.shape[0], size=min(n_local, X_test_proc.shape[0]), replace=False)
        for i, idx in enumerate(idxs, start=1):
            try:
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                    sv[idx,:], feature_names=feature_names, max_display=12, show=False
                )
                plt.tight_layout()
                plt.savefig(outdir / f"shap_waterfall_case_{i}.png", dpi=150)
                plt.close()
            except Exception:
                contrib = pd.Series(np.abs(sv[idx,:]), index=feature_names).sort_values(ascending=False).head(10)
                (outdir / f"shap_case_{i}_top10.txt").write_text(contrib.to_string(), encoding="utf-8")

        return {"shap_disponible": True}
    except Exception:
        return {"shap_disponible": False}

# ------------------------------------------------------------------------------------
# 6) lime: explicaciones locales en los mismos 3 casos de shap
# - trabaja sobre el espacio numerico transformado con predict_proba del modelo
# ------------------------------------------------------------------------------------
def explicar_lime(X_train_proc, X_test_proc, feature_names, proba_fn, outdir: Path, idxs=None):
    LimeTabularExplainer = try_import_lime()
    if LimeTabularExplainer is None:
        return {"lime_disponible": False}

    explainer = LimeTabularExplainer(
        X_train_proc,
        feature_names=feature_names,
        class_names=["no_enfermedad","enfermedad"],
        discretize_continuous=True,
        random_state=SEED
    )

    if idxs is None:
        idxs = np.random.choice(X_test_proc.shape[0], size=min(3, X_test_proc.shape[0]), replace=False)

    for i, idx in enumerate(idxs, start=1):
        exp = explainer.explain_instance(X_test_proc[idx], proba_fn, num_features=10)
        txt = "\n".join([f"{feat}: {weight:.4f}" for feat, weight in exp.as_list()])
        (outdir / f"lime_case_{i}.txt").write_text(txt, encoding="utf-8")

    return {"lime_disponible": True, "idxs": idxs.tolist()}

# ------------------------------------------------------------------------------------
# 7) analisis de sesgo y etica
# - se evalua desempeño por subgrupos sensibles (sexo, grupo etario)
# - se guarda csv con precision, recall, f1 por subgrupo
# ------------------------------------------------------------------------------------
def evaluar_sesgo(X_test_raw: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, outdir: Path):
    df_eval = X_test_raw.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred

    cols = {c: c.lower() for c in df_eval.columns}
    df_eval.rename(columns=cols, inplace=True)

    grupos = []
    if "sex" in df_eval.columns:
        grupos.append(("sex", df_eval["sex"].astype(str)))
    if "age" in df_eval.columns:
        edades = pd.to_numeric(df_eval["age"], errors="coerce")
        bins = [0, 39, 54, 120]
        labels = ["<=39", "40-54", "55+"]
        grupos.append(("age_group", pd.cut(edades, bins=bins, labels=labels, include_lowest=True)))

    filas = []
    for nombre, serie in grupos:
        for g, sub in df_eval.groupby(serie):
            if sub.shape[0] == 0:
                continue
            acc = accuracy_score(sub["y_true"], sub["y_pred"])
            prec = precision_score(sub["y_true"], sub["y_pred"], zero_division=0)
            rec = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)
            f1 = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
            filas.append({
                "grupo": f"{nombre}:{g}",
                "n": int(sub.shape[0]),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            })
    if filas:
        pd.DataFrame(filas).sort_values(["grupo"]).to_csv(outdir / "sesgo_subgrupos.csv", index=False)

# ------------------------------------------------------------------------------------
# 8) script principal
# - orquesta todo el flujo y genera artefactos para el pdf/zip de entrega
# ------------------------------------------------------------------------------------
def main():
    outdir = asegurar_directorio_en_script("resultados_mod9")

    # 1) carga y exploracion
    df, target_col = cargar_heart()
    resumen = {
        "n_filas": int(df.shape[0]),
        "n_columnas": int(df.shape[1]),
        "target": target_col,
        "columnas": df.columns.tolist()
    }
    (outdir / "exploracion_basica.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    # 2) preprocesamiento y split
    X_train_raw, X_test_raw, y_train, y_test, X_train_proc, X_test_proc, pre, info = preparar_datos(df, target_col)

    # 3) modelo (rf por defecto; si quieres lr/xgb cambia 'preferencia' a 'lr' o 'xgb')
    modelo = construir_modelo(input_dim=X_train_proc.shape[1], preferencia="rf")

    # 4) entrenamiento y evaluacion
    metrics, y_pred, proba = entrenar_y_evaluar(modelo, X_train_proc, y_train, X_test_proc, y_test, outdir, "rf")
    pd.DataFrame([metrics]).to_csv(outdir / "metricas_resumen.csv", index=False)

    # 5) shap global + local (3 casos)
    shap_info = explicar_shap(modelo, X_train_proc, X_test_proc, info["feature_names"], outdir, n_local=3)

    # indices para lime (si shap no devolvio indices, se generan nuevos)
    idxs_lime = None
    if isinstance(shap_info, dict) and shap_info.get("shap_disponible", False):
        idxs_lime = np.random.choice(X_test_proc.shape[0], size=min(3, X_test_proc.shape[0]), replace=False)

    # wrapper para lime -> predict_proba en el espacio transformado
    def proba_fn(arr):
        try:
            p = modelo.predict_proba(arr)[:, 1]
            return np.c_[1 - p, p]
        except Exception:
            s = modelo.decision_function(arr)
            p = 1 / (1 + np.exp(-s))
            return np.c_[1 - p, p]

    lime_info = explicar_lime(X_train_proc, X_test_proc, info["feature_names"], proba_fn, outdir, idxs=idxs_lime)

    # 7) sesgos y etica: desempeño por sexo y grupo etario
    evaluar_sesgo(X_test_raw, y_test, y_pred, outdir)

    # 8) breve resumen para readme
    resumen_txt = [
        "# resumen modulo 9",
        "- dataset: heart (kaggle) con limpieza minima y tipificacion basica",
        f"- modelo: random forest, auc={metrics['auc']:.3f}, f1={metrics['f1']:.3f}",
        "- shap: summary dot + bar global; 3 casos locales (waterfall si disponible)",
        "- lime: 3 casos en los mismos indices, comparables con shap",
        "- sesgos: csv con metricas por sexo y grupo etario",
        "- todos los artefactos se guardan en la carpeta 'resultados_mod9' junto al script"
    ]
    (outdir / "README_mod9_resumen.txt").write_text("\n".join(resumen_txt), encoding="utf-8")

    print("listo. artefactos en:", outdir.resolve())
    print("incluye: metricas_resumen.csv, shap_summary_dot.png, shap_summary_bar.png, lime_case_*.txt, sesgo_subgrupos.csv")

if __name__ == "__main__":
    main()