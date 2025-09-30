# actividad sesión 5 — cnn con regularización y optimización (cifar-10 o fashion-mnist)
#
# objetivos:
# - cargar y normalizar un dataset de visión: por defecto **cifar-10**; opcional **fashion-mnist** con --dataset
# - preparar un split de validación a partir de train para monitorear sobreajuste
# - diseñar una **cnn pequeña** con: conv-bn-relu, maxpool, **l2** y **dropout** como regularización
# - entrenar con **adam** o **rmsprop**, usar **early stopping** y **reduceLROnPlateau**
# - registrar y graficar la evolución del **learning rate** por época
# - evaluar en test, generar **matriz de confusión** y **reporte de clasificación**
# - guardar resultados en `resultados_sesion5/` (curvas, lr, matriz, reporte, resumen.json, modelo y resumen de capas)

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# importación de librerías principales
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras import mixed_precision
except Exception as e:
    print("\nerror: no fue posible importar tensorflow/keras.\n"
          "apple silicon:  pip install tensorflow-macos tensorflow-metal\n"
          "otras plataformas: pip install tensorflow\n")
    raise

try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception:
    print("\nerror: falta scikit-learn. instalar con: pip install scikit-learn\n")
    raise

# en apple silicon puede resultar más estable forzar float32
mixed_precision.set_global_policy("float32")

# -----------------------------
# utilidades de carpeta y plots
# -----------------------------

def asegurar_dir_en_script(nombre: str) -> str:
    # crea una carpeta dentro del mismo directorio del archivo actual
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, nombre)
    os.makedirs(out, exist_ok=True)
    return out

def ruta_resultado(nombre: str, outdir: str) -> str:
    # genera una ruta dentro de la carpeta de resultados
    return os.path.join(outdir, nombre)

def fijar_semillas(semilla: int = 42):
    # establece semillas para reproducibilidad
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

def plot_curvas(history: dict, titulo: str, outpath: str):
    # genera curvas de entrenamiento y validación (loss y accuracy)
    plt.figure(figsize=(10, 4))
    # pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.get("loss", []), label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.title("pérdida"); plt.xlabel("época"); plt.ylabel("loss"); plt.legend(loc="best")
    # accuracy
    plt.subplot(1, 2, 2)
    if "accuracy" in history:
        plt.plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val")
    plt.title("precisión (accuracy)"); plt.xlabel("época"); plt.ylabel("accuracy"); plt.legend(loc="best")
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def plot_lr(learning_rates, outpath: str, titulo="learning rate por época"):
    # grafica la evolución del learning rate efectivo
    plt.figure(figsize=(6, 4))
    plt.plot(learning_rates, marker="o")
    plt.title(titulo); plt.xlabel("época"); plt.ylabel("learning rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def plot_matriz_confusion(y_true, y_pred, class_names, outpath: str, titulo="matriz de confusión — test"):
    # dibuja la matriz de confusión con anotaciones en cada celda
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="verdadero", xlabel="predicho", title=titulo)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# carga y preprocesamiento
# -----------------------------

def cargar_dataset(nombre="cifar10"):
    """
    retorna:
      (x_train, y_train), (x_test, y_test), class_names, input_shape, num_clases
    normaliza a [0,1] y entrega tensores float32.
    """
    if nombre.lower() == "fashion_mnist":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
        # fashion-mnist es (H,W) en grises → se expande a (H,W,1) y se replica a 3 canales
        x_tr = np.repeat(x_tr[..., np.newaxis], 3, axis=-1)
        x_te = np.repeat(x_te[..., np.newaxis], 3, axis=-1)
        class_names = ["t_shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]
        input_shape = (28, 28, 3)
        num_clases = 10
    else:
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
        y_tr = y_tr.squeeze(); y_te = y_te.squeeze()
        class_names = ["avión","auto","pájaro","gato","ciervo","perro","rana","caballo","barco","camión"]
        input_shape = (32, 32, 3)
        num_clases = 10

    x_tr = x_tr.astype("float32") / 255.0
    x_te = x_te.astype("float32") / 255.0
    return (x_tr, y_tr), (x_te, y_te), class_names, input_shape, num_clases

# -----------------------------
# arquitectura cnn regularizada
# -----------------------------

def construir_cnn(input_shape, num_clases, l2_lambda=1e-4, dropout_rate=0.3):
    """
    cnn compacta:
      - bloques conv (Conv2D -> BN -> ReLU) + MaxPool
      - regularización L2 en kernels
      - dropout en bloques y cabeza
      - global average pooling + densa final
    """
    l2 = regularizers.l2(l2_lambda) if l2_lambda and l2_lambda > 0 else None

    inp = layers.Input(shape=input_shape)
    x = inp

    # bloque 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # bloque 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # bloque 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_clases, activation="softmax")(x)

    return models.Model(inp, out, name="cnn_regularizada")

# -----------------------------
# optimizadores y callbacks
# -----------------------------

class LRHistory(tf.keras.callbacks.Callback):
    # callback para registrar el learning rate efectivo por época
    def __init__(self):
        super().__init__()
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)

def crear_optimizador(nombre="adam", lr=1e-3, clipnorm=1.0):
    if nombre.lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=clipnorm)
    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

def crear_callbacks(outdir: str, monitor="val_loss"):
    # incluye early stopping, reducción de lr y checkpoint del mejor modelo
    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    lrhist = LRHistory()
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(outdir, "mejor_modelo.keras"),
        monitor=monitor, save_best_only=True
    )
    return [es, rlr, lrhist, ckpt], lrhist

# -----------------------------
# entrenamiento y evaluación
# -----------------------------

def entrenar_y_evaluar(dataset="cifar10",
                       optimizador="adam",
                       lr=1e-3,
                       l2_lambda=1e-4,
                       dropout_rate=0.3,
                       batch=128,
                       epocas=30,
                       val_split=0.1):
    outdir = asegurar_dir_en_script("resultados_sesion5")
    fijar_semillas(42)

    # datos
    (x_tr, y_tr), (x_te, y_te), class_names, input_shape, num_clases = cargar_dataset(dataset)

    # validación
    n_val = int(val_split * len(x_tr))
    x_val, y_val = x_tr[:n_val], y_tr[:n_val]
    x_tr, y_tr = x_tr[n_val:], y_tr[n_val:]

    # modelo
    modelo = construir_cnn(input_shape, num_clases, l2_lambda=l2_lambda, dropout_rate=dropout_rate)
    opt = crear_optimizador(optimizador, lr=lr, clipnorm=1.0)
    modelo.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # guardado de resumen de arquitectura
    lines = []
    modelo.summary(print_fn=lambda s: lines.append(s))
    with open(ruta_resultado("modelo_resumen.txt", outdir), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # entrenamiento
    cbs, lrhist = crear_callbacks(outdir, monitor="val_loss")
    t0 = time.time()
    hist = modelo.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=epocas,
        batch_size=batch,
        callbacks=cbs,
        verbose=2
    )
    t1 = time.time()

    # curvas y learning rate
    plot_curvas(hist.history, f"cnn regularizada — {dataset}", ruta_resultado("curvas_entrenamiento.png", outdir))
    plot_lr(lrhist.lrs, ruta_resultado("learning_rate.png", outdir))

    # evaluación en test
    test_loss, test_acc = modelo.evaluate(x_te, y_te, verbose=0)
    y_pred = np.argmax(modelo.predict(x_te, verbose=0), axis=1)

    # matriz y reporte
    plot_matriz_confusion(y_te, y_pred, class_names, ruta_resultado("matriz_confusion.png", outdir))
    reporte_txt = classification_report(y_te, y_pred, target_names=class_names, digits=4)
    with open(ruta_resultado("reporte_clasificacion.txt", outdir), "w", encoding="utf-8") as f:
        f.write(reporte_txt)

    # resumen json
    resumen = {
        "dataset": dataset,
        "optimizador": optimizador,
        "learning_rate_inicial": lr,
        "l2_lambda": l2_lambda,
        "dropout_rate": dropout_rate,
        "batch": batch,
        "epocas_max": epocas,
        "val_split": val_split,
        "duracion_s": round(float(t1 - t0), 2),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "learning_rates": lrhist.lrs
    }
    with open(ruta_resultado("resumen.json", outdir), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # guardado del modelo final
    modelo.save(os.path.join(outdir, "modelo_cnn.keras"))

    print("\nresultados en:", outdir)
    print("- curvas_entrenamiento.png")
    print("- learning_rate.png")
    print("- matriz_confusion.png")
    print("- reporte_clasificacion.txt")
    print("- resumen.json")
    print("- modelo_cnn.keras")
    print("- modelo_resumen.txt\n")

    return outdir, resumen

# -----------------------------
# main (argumentos por CLI)
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="sesión 5 — cnn con regularización y optimización")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fashion_mnist"],
                        help="dataset a usar (cifar10 | fashion_mnist)")
    parser.add_argument("--optimizador", type=str, default="adam", choices=["adam", "rmsprop"],
                        help="optimizador (adam | rmsprop)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate inicial")
    parser.add_argument("--l2", type=float, default=1e-4, help="lambda L2 (0 para desactivar)")
    parser.add_argument("--dropout", type=float, default=0.3, help="tasa de dropout")
    parser.add_argument("--batch", type=int, default=128, help="tamaño de batch")
    parser.add_argument("--epocas", type=int, default=30, help="máximo de épocas")
    parser.add_argument("--val_split", type=float, default=0.1, help="proporción de validación de train")
    args = parser.parse_args()

    entrenar_y_evaluar(dataset=args.dataset,
                       optimizador=args.optimizador,
                       lr=args.lr,
                       l2_lambda=args.l2,
                       dropout_rate=args.dropout,
                       batch=args.batch,
                       epocas=args.epocas,
                       val_split=args.val_split)

if __name__ == "__main__":
    main()