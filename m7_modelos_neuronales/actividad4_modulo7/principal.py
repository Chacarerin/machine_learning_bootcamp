# actividad sesión 4 — transfer learning con efficientnetb0 o resnet50 sobre cifar-10
# este script entrena, evalúa y guarda curvas, matriz de confusión y predicciones.


import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# cargar tensorflow/keras (con política float32 para apple silicon) y sklearn
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0, ResNet50
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre
    from tensorflow.keras.applications.resnet import preprocess_input as resnet_pre
    from tensorflow.keras import mixed_precision
    from tensorflow.keras import backend as K
except Exception as e:
    print("\nno fue posible importar tensorflow/keras.\n"
          "sugerencia (apple silicon): pip install tensorflow-macos tensorflow-metal\n"
          "otras plataformas: pip install tensorflow\n"
          f"error: {repr(e)}\n")
    sys.exit(1)

try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception as e:
    print("\nno fue posible importar scikit-learn (sklearn).\n"
          "instalar con: pip install scikit-learn\n"
          f"error: {repr(e)}\n")
    sys.exit(1)

# en apple silicon suele ser más estable usar float32
mixed_precision.set_global_policy("float32")
# nos aseguramos de usar formato de datos 'channels_last' (rgb en el último eje)
K.set_image_data_format("channels_last")

# ------------------------------------------------------------
# utilidades generales
# ------------------------------------------------------------

def asegurar_dir(path):
    # crea carpeta si no existe
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def ruta_resultado(nombre, base_dir):
    # ruta absoluta dentro de la carpeta de resultados
    return os.path.join(base_dir, nombre)

def fijar_semillas(semilla=42):
    # semilla para reproducibilidad razonable
    tf.random.set_seed(semilla)
    np.random.seed(semilla)

def guardar_texto(texto, ruta_txt):
    # guarda texto plano
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(texto)

def plot_curvas(hist, ruta_png, titulo="curvas de entrenamiento"):
    # guarda loss y accuracy para train/val
    plt.figure(figsize=(10,4))
    # loss
    plt.subplot(1,2,1)
    plt.plot(hist.history.get("loss", []), label="train")
    if "val_loss" in hist.history:
        plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("época"); plt.ylabel("loss"); plt.title("pérdida"); plt.legend()
    # accuracy
    plt.subplot(1,2,2)
    if "accuracy" in hist.history:
        plt.plot(hist.history["accuracy"], label="train")
    if "val_accuracy" in hist.history:
        plt.plot(hist.history["val_accuracy"], label="val")
    plt.xlabel("época"); plt.ylabel("accuracy"); plt.title("exactitud"); plt.legend()
    plt.suptitle(titulo, y=1.02)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_matriz_confusion(y_true, y_pred, class_names, ruta_png, titulo="matriz de confusión"):
    # dibuja matriz de confusión como imagen
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="verdadero", xlabel="predicho", title=titulo)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # anota valores
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_predicciones_grid(imagenes, y_true, y_pred, class_names, ruta_png, filas=3, cols=6):
    # muestra una grilla de imágenes con etiqueta real y predicha
    total = filas * cols
    idx = np.random.choice(len(imagenes), total, replace=False)
    plt.figure(figsize=(cols*2.0, filas*2.2))
    for i, k in enumerate(idx):
        plt.subplot(filas, cols, i+1)
        plt.imshow(imagenes[k].astype("uint8"))
        r = class_names[y_true[k]]
        p = class_names[y_pred[k]]
        plt.title(f"r:{r} / p:{p}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# datos (cifar-10 por defecto)
# ------------------------------------------------------------

def cargar_cifar10():
    # carga cifar-10 (50k train, 10k test), imágenes 32x32x3, 10 clases
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    class_names = ["avión","auto","pájaro","gato","ciervo","perro","rana","caballo","barco","camión"]
    return (x_train, y_train), (x_test, y_test), class_names

def crear_datasets_efficientnet(x_train, y_train, x_val, y_val, x_test, y_test, batch=64, imagen_size=224):
    # reescalamos a 224x224 y aplicamos preprocesamiento específico de efficientnet
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    def _resize_preprocess(x, y):
        x = tf.image.resize(x, (imagen_size, imagen_size))
        x = effnet_pre(x)  # normalización según efficientnet
        return x, y

    def _resize_only(x, y):
        x = tf.image.resize(x, (imagen_size, imagen_size))
        x = effnet_pre(x)
        return x, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(batch)
        .map(_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch)
        .map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch)
        .map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds

def crear_datasets_resnet(x_train, y_train, x_val, y_val, x_test, y_test, batch=64, imagen_size=224):
    # lo mismo pero usando el preprocesamiento de resnet
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    def _resize_preprocess(x, y):
        x = tf.image.resize(x, (imagen_size, imagen_size))
        x = resnet_pre(x)
        return x, y

    def _resize_only(x, y):
        x = tf.image.resize(x, (imagen_size, imagen_size))
        x = resnet_pre(x)
        return x, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(batch)
        .map(_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch)
        .map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch)
        .map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds

# ------------------------------------------------------------
# modelo (transfer learning)
# ------------------------------------------------------------

def construir_modelo(base_modelo="efficientnet", num_clases=10, imagen_size=224, train_base=False):
    """
    construye el modelo con un backbone preentrenado en imagenet.
    solución robusta para el error de shape mismatch:
      - forzamos channels_last
      - usamos input_tensor explícito con 3 canales (rgb)
      - si cargar pesos imagenet falla, hacemos fallback a weights=None
    """
    # aseguramos canales al final (rgb)
    K.set_image_data_format("channels_last")

    # definimos un input_tensor rgb explícito
    input_tensor = layers.Input(shape=(imagen_size, imagen_size, 3))

    try:
        if base_modelo == "resnet":
            base = ResNet50(include_top=False, weights="imagenet",
                            input_tensor=input_tensor, pooling="avg")
        else:
            base = EfficientNetB0(include_top=False, weights="imagenet",
                                  input_tensor=input_tensor, pooling="avg")
    except ValueError as e:
        # si hay choque de canales con los pesos preentrenados (p. ej. (3,3,1,32) vs (3,3,3,32))
        print("\n[aviso] no se pudieron cargar pesos imagenet por ‘shape mismatch’. "
              "reintentando con weights=None (entrenar desde cero la base)...\n")
        if base_modelo == "resnet":
            base = ResNet50(include_top=False, weights=None,
                            input_tensor=input_tensor, pooling="avg")
        else:
            base = EfficientNetB0(include_top=False, weights=None,
                                  input_tensor=input_tensor, pooling="avg")

    # por defecto congelamos el backbone; si se activa fine-tuning lo destrabamos parcialmente
    base.trainable = train_base

    x = base.output
    x = layers.Dropout(0.25)(x)
    salidas = layers.Dense(num_clases, activation="softmax")(x)
    modelo = models.Model(inputs=input_tensor, outputs=salidas, name=f"tl_{base_modelo}")

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    modelo.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo, base

# ------------------------------------------------------------
# entrenamiento y evaluación
# ------------------------------------------------------------

def entrenar_y_evaluar(base_modelo="efficientnet", imagen_size=224, batch=64, epocas=10, fine_tune=False):
    # carpeta de resultados y semillas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "resultados_sesion4")
    asegurar_dir(out_dir)
    fijar_semillas(42)

    # datos y split train/val/test
    (x_train, y_train), (x_test, y_test), class_names = cargar_cifar10()
    n_val = int(0.1 * len(x_train))
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    # datasets tf.data según backbone
    if base_modelo == "resnet":
        train_ds, val_ds, test_ds = crear_datasets_resnet(
            x_train, y_train, x_val, y_val, x_test, y_test, batch, imagen_size
        )
    else:
        train_ds, val_ds, test_ds = crear_datasets_efficientnet(
            x_train, y_train, x_val, y_val, x_test, y_test, batch, imagen_size
        )

    # modelo (fase 1: entrenar solo la cabeza)
    modelo, backbone = construir_modelo(
        base_modelo, num_clases=len(class_names), imagen_size=imagen_size, train_base=False
    )

    # callbacks: early stopping + reducción de lr si se estanca
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6)

    hist = modelo.fit(train_ds, validation_data=val_ds, epochs=epocas, callbacks=[es, rlr], verbose=2)
    plot_curvas(hist, ruta_resultado("curvas_entrenamiento_fase1.png", out_dir),
                f"{base_modelo} — fase 1 (cabeza)")

    # opcional: fine-tuning (destrabar último tercio del backbone)
    if fine_tune:
        backbone.trainable = True
        limite = max(1, int(len(backbone.layers) * 0.67))  # bloqueamos ~2/3 iniciales
        for layer in backbone.layers[:limite]:
            layer.trainable = False

        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        hist2 = modelo.fit(train_ds, validation_data=val_ds, epochs=max(2, epocas // 2), callbacks=[es, rlr], verbose=2)
        plot_curvas(hist2, ruta_resultado("curvas_entrenamiento_fase2.png", out_dir),
                    f"{base_modelo} — fase 2 (fine-tuning)")

    # evaluación en test
    test_loss, test_acc = modelo.evaluate(test_ds, verbose=0)

    # predicciones para matriz de confusión y grilla visual
    # guardamos también x_test reescalado a 224 para mostrar imágenes grandes en la grilla
    x_test_big = tf.image.resize(x_test, (imagen_size, imagen_size)).numpy().astype("uint8")
    y_pred = np.argmax(modelo.predict(test_ds, verbose=0), axis=1)

    # matriz de confusión y grilla de ejemplos
    plot_matriz_confusion(y_test, y_pred, class_names,
                          ruta_resultado("matriz_confusion.png", out_dir),
                          titulo=f"matriz de confusión — {base_modelo}")
    plot_predicciones_grid(x_test_big, y_test, y_pred, class_names,
                           ruta_resultado("predicciones_vs_reales.png", out_dir))

    # classification report a txt
    reporte = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    guardar_texto(reporte, ruta_resultado("reporte_clasificacion.txt", out_dir))

    # resumen json
    resumen = {
        "modelo_base": base_modelo,
        "imagen_size": imagen_size,
        "batch": batch,
        "epocas": epocas,
        "fine_tune": fine_tune,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    }
    guardar_texto(json.dumps(resumen, indent=2, ensure_ascii=False),
                  ruta_resultado("resumen.json", out_dir))

    # guardo modelo y resumen de capas
    modelo.save(os.path.join(out_dir, f"modelo_{base_modelo}.keras"))
    lines = []
    modelo.summary(print_fn=lambda s: lines.append(s))
    guardar_texto("\n".join(lines), ruta_resultado(f"modelo_{base_modelo}_resumen.txt", out_dir))

    # mensaje final
    print("\nlisto. resultados en:", out_dir)
    print("- curvas_entrenamiento_fase1.png")
    if fine_tune:
        print("- curvas_entrenamiento_fase2.png")
    print("- matriz_confusion.png")
    print("- predicciones_vs_reales.png")
    print("- reporte_clasificacion.txt")
    print("- resumen.json")
    print(f"- modelo_{base_modelo}.keras")
    return out_dir, resumen

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    # flags para elegir backbone, tamaño, batch, épocas y activar fine-tuning
    parser = argparse.ArgumentParser(description="actividad sesión 4 — transfer learning")
    parser.add_argument("--modelo", type=str, default="efficientnet", choices=["efficientnet", "resnet"],
                        help="selecciona backbone preentrenado (efficientnet | resnet)")
    parser.add_argument("--imagen_size", type=int, default=224, help="tamaño a reescalar las imágenes")
    parser.add_argument("--batch", type=int, default=64, help="tamaño de batch")
    parser.add_argument("--epocas", type=int, default=10, help="épocas de entrenamiento en fase 1")
    parser.add_argument("--fine_tune", action="store_true", help="activar fine-tuning (fase 2)")
    args = parser.parse_args()

    entrenar_y_evaluar(base_modelo=args.modelo,
                       imagen_size=args.imagen_size,
                       batch=args.batch,
                       epocas=args.epocas,
                       fine_tune=args.fine_tune)

if __name__ == "__main__":
    main()