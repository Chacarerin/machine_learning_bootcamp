# actividad sesion 1 – clasificador de imagenes con redes neuronales (fashion mnist)
#
# objetivos:
# - cargar y normalizar fashion_mnist
# - convertir etiquetas a one-hot
# - diseñar una red densa: flatten -> dense(relu) -> dense(tanh) -> softmax(10)
# - entrenar probando combinaciones de perdida (cce, mse) y optimizador (adam, sgd)
# - graficar curvas de entrenamiento/validacion y evaluar en test
# - guardar resumen comparativo en json/txt manteniendo la misma estructura de trabajo

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# intento de import de tensorflow con salida clara si no esta instalado
try:
    import tensorflow as tf
except Exception as e:
    print("error: tensorflow no esta instalado. instale con 'pip install tensorflow'.")
    raise

from sklearn.metrics import classification_report, confusion_matrix

# reproducibilidad simple
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# ---------------------------------
# utilidades de carpeta y graficos
# ---------------------------------

def asegurar_directorio_en_script(nombre: str) -> Path:
    # crea una carpeta dentro del mismo directorio del archivo actual
    base = Path(os.path.dirname(__file__))
    out = base / nombre
    out.mkdir(parents=True, exist_ok=True)
    return out

def plot_curvas(history: dict, titulo: str, outpath: Path):
    # genera una figura con dos subplots: perdida y accuracy
    plt.figure(figsize=(8, 4.5))

    # subplot 1: perdida
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.title("perdida")
    plt.xlabel("epoca")
    plt.ylabel("loss")
    plt.legend(loc="best")

    # subplot 2: accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val")
    plt.title("precision (accuracy)")
    plt.xlabel("epoca")
    plt.ylabel("accuracy")
    plt.legend(loc="best")

    plt.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# -----------------------------
# carga y preprocesamiento
# -----------------------------

def cargar_y_preprocesar():
    # carga fashion mnist, normaliza a [0,1] y convierte y a one-hot
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x_tr = x_tr.astype("float32") / 255.0
    x_te = x_te.astype("float32") / 255.0

    # one-hot
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes=10)
    y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes=10)

    return (x_tr, y_tr, y_tr_oh), (x_te, y_te, y_te_oh)

# -----------------------------
# definicion del modelo
# -----------------------------

def construir_modelo():
    # red densa segun enunciado: flatten -> dense(relu) -> dense(tanh) -> softmax(10)
    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return modelo

def compilar_modelo(modelo, loss_name: str, opt_name: str):
    # crea el optimizador y compila con la perdida seleccionada
    if opt_name == "adam":
        opt = tf.keras.optimizers.Adam()
    elif opt_name == "sgd":
        opt = tf.keras.optimizers.SGD()
    else:
        raise ValueError("optimizador no soportado")

    modelo.compile(
        optimizer=opt,
        loss=loss_name,
        metrics=["accuracy"],
    )
    return modelo

# -----------------------------
# entrenamiento y evaluacion
# -----------------------------

def entrenar_y_evaluar(x_tr, y_tr_oh, x_te, y_te_oh, y_te, loss_name, opt_name, outdir: Path, epocas=8):
    # construye, compila, entrena y evalua una combinacion (loss, opt)
    modelo = construir_modelo()
    modelo = compilar_modelo(modelo, loss_name, opt_name)

    hist = modelo.fit(
        x_tr, y_tr_oh,
        epochs=epocas,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    # graficos
    titulo = f"loss={loss_name} | opt={opt_name}"
    plot_curvas(hist.history, titulo, outdir / f"curvas_{loss_name}_{opt_name}.png")

    # evaluacion en test
    test_loss, test_acc = modelo.evaluate(x_te, y_te_oh, verbose=0)

    # predicciones y reporte de clasificacion
    y_pred_probs = modelo.predict(x_te, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    reporte = classification_report(y_te, y_pred, digits=4)
    cm = confusion_matrix(y_te, y_pred)

    # guardo txt con resultados de la combinacion
    with open(outdir / f"reporte_{loss_name}_{opt_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"# combinacion: loss={loss_name}, opt={opt_name}\n")
        f.write(f"test_accuracy: {test_acc:.4f}\n")
        f.write(f"test_loss: {test_loss:.4f}\n\n")
        f.write("classification_report:\n")
        f.write(reporte + "\n")
        f.write("confusion_matrix:\n")
        f.write(np.array2string(cm))

    # retorno resumen para comparacion global
    resumen = {
        "loss": loss_name,
        "optimizer": opt_name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "history": {
            "accuracy": float(hist.history["accuracy"][-1]),
            "val_accuracy": float(hist.history.get("val_accuracy", [np.nan])[-1]),
            "loss": float(hist.history["loss"][-1]),
            "val_loss": float(hist.history.get("val_loss", [np.nan])[-1]),
        },
        "artefactos": {
            "curvas_png": f"curvas_{loss_name}_{opt_name}.png",
            "reporte_txt": f"reporte_{loss_name}_{opt_name}.txt"
        }
    }
    return resumen

# -----------------------------
# flujo principal de la sesion
# -----------------------------

def main():
    outdir = asegurar_directorio_en_script("resultados_sesion1")

    # 1) datos
    (x_tr, y_tr, y_tr_oh), (x_te, y_te, y_te_oh) = cargar_y_preprocesar()

    # 2) grid simple de perdidas y optimizadores
    perdidas = ["categorical_crossentropy", "mse"]
    optimizadores = ["adam", "sgd"]

    # 3) entreno todas las combinaciones
    comparacion = []
    for loss_name in perdidas:
        for opt_name in optimizadores:
            print(f"entrenando: loss={loss_name} | opt={opt_name}")
            res = entrenar_y_evaluar(
                x_tr, y_tr_oh,
                x_te, y_te_oh, y_te,
                loss_name, opt_name,
                outdir, epocas=8
            )
            comparacion.append(res)

    # 4) elijo la mejor por accuracy en test
    mejor = max(comparacion, key=lambda r: r["test_accuracy"])
    detalle = {
        "dataset": "fashion_mnist",
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "input_shape": list(x_tr.shape[1:]),
        "modelo": "flatten -> dense(256, relu) -> dropout(0.2) -> dense(128, tanh) -> dropout(0.2) -> dense(10, softmax)",
        "combinaciones": comparacion,
        "mejor": mejor,
        "criterio_mejora": "max test_accuracy"
    }

    # 5) guardo resumen json y txt
    with open(outdir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(detalle, f, ensure_ascii=False, indent=2)

    with open(outdir / "resumen.txt", "w", encoding="utf-8") as f:
        f.write("actividad sesion 1 – clasificador denso (fashion mnist)\n\n")
        f.write(f"mejor combinacion: loss={mejor['loss']}, opt={mejor['optimizer']}\n")
        f.write(f"test_accuracy: {mejor['test_accuracy']:.4f}\n")
        f.write(f"test_loss: {mejor['test_loss']:.4f}\n")
        f.write("\n")
        f.write("nota: las curvas y reportes por combinacion se guardan en archivos separados.\n")

    print("listo. resultados guardados en:", outdir.resolve())
    print("- curvas_*.png")
    print("- reporte_*.txt")
    print("- resumen.json")
    print("- resumen.txt")

if __name__ == "__main__":
    main()