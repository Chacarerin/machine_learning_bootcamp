# actividad sesion 2 – rnn (lstm) para imdb + gan basica para mnist
#
# objetivos:
# - parte a: clasificar sentimiento con una lstm sobre imdb
# - parte b: entrenar una gan simple para generar digitos (mnist)
# todos los resultados se guardan en 'resultados_sesion2/' junto a este script
## para ejecutar: python principal.py
# requiere tensorflow (>=2.0)
# puede tomar varios minutos dependiendo de tu hardware
import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# fijar semillas para reproducibilidad basica
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# utilidades generales
# -----------------------------

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
    plt.plot(history.get("loss", []), label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.title("perdida")
    plt.xlabel("epoca")
    plt.ylabel("loss")
    plt.legend(loc="best")

    # subplot 2: accuracy
    plt.subplot(1, 2, 2)
    if "accuracy" in history:
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

def escribir_txt(path: Path, lineas):
    # escribe lista de lineas de texto en un archivo
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lineas, (list, tuple)):
            for l in lineas:
                f.write(str(l) + "\n")
        else:
            f.write(str(lineas) + "\n")

# -----------------------------
# parte a: imdb con lstm
# -----------------------------

def cargar_y_preprocesar_imdb(vocab_size=20000, maxlen=200):
    # carga imdb tokenizado por indices de frecuencia y aplica padding
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    x_tr = pad_sequences(x_tr, maxlen=maxlen, padding="post", truncating="post")
    x_te = pad_sequences(x_te, maxlen=maxlen, padding="post", truncating="post")
    return (x_tr, y_tr), (x_te, y_te)

def construir_lstm_imdb(vocab_size=20000, maxlen=200, embedding_dim=64, lstm_units=64):
    # embedding -> lstm -> densa sigmoid
    modelo = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        layers.LSTM(lstm_units),
        layers.Dense(1, activation="sigmoid")
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modelo

def entrenar_y_evaluar_imdb(outdir: Path, epocas=3, batch_size=128):
    # entrena lstm, guarda curvas, matriz de confusion y resumen
    (x_tr, y_tr), (x_te, y_te) = cargar_y_preprocesar_imdb()
    modelo = construir_lstm_imdb()

    hist = modelo.fit(
        x_tr, y_tr,
        validation_split=0.2,
        epochs=epocas,
        batch_size=batch_size,
        verbose=1
    )

    # graficas de entrenamiento
    plot_curvas(hist.history, "imdb – lstm", outdir / "imdb_curvas.png")

    # evaluacion sobre test
    loss_te, acc_te = modelo.evaluate(x_te, y_te, verbose=0)

    # predicciones y matriz de confusion sin sklearn
    y_prob = modelo.predict(x_te, batch_size=batch_size, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    tn = int(np.sum((y_te == 0) & (y_pred == 0)))
    fp = int(np.sum((y_te == 0) & (y_pred == 1)))
    fn = int(np.sum((y_te == 1) & (y_pred == 0)))
    tp = int(np.sum((y_te == 1) & (y_pred == 1)))

    cm = np.array([[tn, fp], [fn, tp]])

    # heatmap simple
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("matriz de confusion imdb")
    plt.colorbar()
    plt.xticks([0, 1], ["neg", "pos"])
    plt.yticks([0, 1], ["neg", "pos"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("prediccion")
    plt.ylabel("real")
    plt.tight_layout()
    plt.savefig(outdir / "imdb_matriz_confusion.png", dpi=150)
    plt.close()

    # guardar resumen parte a
    resumen_a = {
        "parte": "A",
        "modelo": "embedding -> lstm(64) -> dense(1, sigmoid)",
        "test_loss": float(loss_te),
        "test_accuracy": float(acc_te),
        "matriz_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "notas": "maxlen=200; embedding_dim=64; lstm_units=64; optimizador=adam; loss=binary_crossentropy"
    }
    escribir_txt(outdir / "imdb_resumen.txt", [
        "parte a – imdb con lstm",
        f"test_accuracy: {acc_te:.4f}",
        f"test_loss: {loss_te:.4f}",
        f"matriz_confusion: tn={tn}, fp={fp}, fn={fn}, tp={tp}",
        "graficos: imdb_curvas.png, imdb_matriz_confusion.png"
    ])

    return resumen_a

# -----------------------------
# parte b: gan basica para mnist
# -----------------------------

def cargar_mnist():
    # carga mnist y normaliza a [-1, 1] para usar con tanh
    (x_tr, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_tr = (x_tr.astype("float32") / 127.5) - 1.0  # [0,255] -> [-1,1]
    x_tr = np.expand_dims(x_tr, -1)  # (n, 28, 28, 1)
    return x_tr

def construir_generador(z_dim=100):
    # generador simple tipo mlp que produce 28x28 con tanh
    modelo = models.Sequential([
        layers.Input(shape=(z_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(28 * 28, activation="tanh"),
        layers.Reshape((28, 28, 1))
    ])
    return modelo

def construir_discriminador():
    # discriminador simple tipo mlp sobre imagen 28x28
    modelo = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
    return modelo

def guardar_grid_imagenes(imgs, path: Path, filas=4, cols=4):
    # imgs se asume en rango [-1, 1]; se reescala a [0,1] para guardar
    imgs = (imgs + 1.0) / 2.0
    plt.figure(figsize=(cols * 2.0, filas * 2.0))
    for i in range(filas * cols):
        plt.subplot(filas, cols, i + 1)
        if i < imgs.shape[0]:
            plt.imshow(imgs[i, ..., 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def entrenar_gan_mnist(outdir: Path, iteraciones=3000, z_dim=100, batch_size=128, guardar_cada=500):
    # entrenamiento alternado de generador y discriminador; guarda grids periodicamente
    x_tr = cargar_mnist()
    n = x_tr.shape[0]

    generador = construir_generador(z_dim=z_dim)
    discriminador = construir_discriminador()

    # modelo combinado g(z)->d(g(z))
    discriminador.trainable = False
    z_input = layers.Input(shape=(z_dim,))
    img_fake = generador(z_input)
    val_fake = discriminador(img_fake)
    combinado = models.Model(z_input, val_fake)
    combinado.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                      loss="binary_crossentropy")

    # contenedores de historial
    hist = {"d_loss": [], "d_acc": [], "g_loss": []}

    # bucle de entrenamiento
    for it in range(1, iteraciones + 1):
        # 1) entrenar discriminador con batch real + fake
        idx = np.random.randint(0, n, batch_size)
        reales = x_tr[idx]

        z = np.random.normal(size=(batch_size, z_dim))
        falsas = generador.predict(z, verbose=0)

        x = np.concatenate([reales, falsas], axis=0)
        y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)

        d_metrics = discriminador.train_on_batch(x, y)
        d_loss, d_acc = float(d_metrics[0]), float(d_metrics[1])

        # 2) entrenar generador via combinado intentando engañar al discriminador
        z = np.random.normal(size=(batch_size, z_dim))
        y_gen = np.ones((batch_size, 1))
        g_loss = float(combinado.train_on_batch(z, y_gen))

        # guardar historial
        hist["d_loss"].append(d_loss)
        hist["d_acc"].append(d_acc)
        hist["g_loss"].append(g_loss)

        # guardar muestras periodicas
        if it % guardar_cada == 0 or it == 1:
            z = np.random.normal(size=(16, z_dim))
            muestras = generador.predict(z, verbose=0)
            guardar_grid_imagenes(muestras, outdir / f"gan_iter_{it:04d}.png", filas=4, cols=4)
            print(f"iter {it:04d} | d_loss={d_loss:.3f} d_acc={d_acc:.3f} | g_loss={g_loss:.3f}")

    # graficos de perdidas y accuracy del discriminador
    plt.figure(figsize=(8, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(hist["d_loss"], label="d_loss")
    plt.plot(hist["g_loss"], label="g_loss")
    plt.title("perdida gan")
    plt.xlabel("iteracion")
    plt.ylabel("loss")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(hist["d_acc"], label="d_accuracy")
    plt.title("accuracy discriminador")
    plt.xlabel("iteracion")
    plt.ylabel("accuracy")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outdir / "gan_curvas.png", dpi=150)
    plt.close()

    # resumen parte b
    resumen_b = {
        "parte": "B",
        "arquitectura": {
            "generador": "dense(256,relu)->dense(512,relu)->dense(784,tanh)->reshape(28,28,1)",
            "discriminador": "flatten->dense(512,relu)->dense(256,relu)->dense(1,sigmoid)"
        },
        "iteraciones": iteraciones,
        "batch_size": batch_size,
        "guardar_cada": guardar_cada,
        "archivos_muestras": [f"gan_iter_{i:04d}.png" for i in range(1, iteraciones + 1) if (i % guardar_cada == 0 or i == 1)],
        "notas": "las imagenes generadas se guardan periodicamente; inspeccionar visualmente para ver cuando comienzan a parecer digitos."
    }
    escribir_txt(outdir / "gan_resumen.txt", [
        "parte b – gan basica para mnist",
        f"iteraciones: {iteraciones}",
        f"batch_size: {batch_size}",
        "graficos: gan_curvas.png y gan_iter_*.png"
    ])

    return resumen_b, hist

# -----------------------------
# main
# -----------------------------

def main():
    # 1) carpeta de resultados junto al script
    outdir = asegurar_directorio_en_script("resultados_sesion2")

    # 2) parte a: imdb + lstm
    resumen_a = entrenar_y_evaluar_imdb(outdir, epocas=3, batch_size=128)

    # 3) parte b: gan (mnist)
    resumen_b, hist_b = entrenar_gan_mnist(outdir, iteraciones=3000, z_dim=100, batch_size=128, guardar_cada=500)

    # 4) consolidar resumen general (json + txt)
    resumen = {
        "actividad": "sesion 2 – rnn (imdb) + gan (mnist)",
        "parte_a": resumen_a,
        "parte_b": resumen_b
    }
    with open(outdir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    escribir_txt(outdir / "resumen.txt", [
        "actividad sesion 2 – rnn (lstm) y gan",
        "",
        "parte a – imdb:",
        f"- test_accuracy: {resumen_a['test_accuracy']:.4f}",
        f"- test_loss: {resumen_a['test_loss']:.4f}",
        f"- matriz_confusion: {resumen_a['matriz_confusion']}",
        "graficos: imdb_curvas.png, imdb_matriz_confusion.png",
        "",
        "parte b – gan mnist:",
        f"- iteraciones: {resumen_b['iteraciones']}",
        f"- batch_size: {resumen_b['batch_size']}",
        "graficos: gan_curvas.png y gan_iter_*.png",
        "",
        "nota: esta actividad usa tf.keras.datasets (imdb, mnist) y guarda todos los resultados en 'resultados_sesion2/'."
    ])

    # 5) imprimir recordatorio de archivos
    print("listo. resultados guardados en:", outdir.resolve())
    print("- imdb_curvas.png, imdb_matriz_confusion.png, imdb_resumen.txt")
    print("- gan_curvas.png, gan_iter_*.png, gan_resumen.txt")
    print("- resumen.json, resumen.txt")

if __name__ == "__main__":
    main()