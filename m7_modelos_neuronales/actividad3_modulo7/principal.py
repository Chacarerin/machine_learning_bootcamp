# ACTIVIDAD SESIÓN 3 — AUTOENCODERS (RECONSTRUCCIÓN Y DENOISING) CON MNIST
# objetivo: entrenar dos autoencoders sencillos (básico y denoising) sobre mnist,
#           guardar curvas, métricas y visualizaciones en 'resultados_sesion3/'.

import os
import sys
import json
import time
import argparse
import numpy as np

# intento cargar tensorflow y los módulos necesarios.
# si fallara la importación, muestro un mensaje claro de instalación.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras import mixed_precision
except Exception as e:
    msg = (
        "\nno fue posible importar tensorflow.\n"
        "en mac con apple silicon:\n"
        "  pip install tensorflow-macos tensorflow-metal\n"
        "otras plataformas:\n"
        "  pip install tensorflow\n\n"
        f"error: {repr(e)}\n"
    )
    print(msg)
    sys.exit(1)

# forzamos política de precisión en float32.
# comentario: en apple silicon con backend metal esto ayuda a evitar inestabilidades
# numéricas (sobre todo con binary crossentropy) y hace el entrenamiento más estable.
mixed_precision.set_global_policy("float32")

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# UTILIDADES GENERALES
# ------------------------------------------------------------

def asegurar_directorio(ruta_dir):
    # crea la carpeta si no existe; útil para guardar resultados ordenados.
    if not os.path.exists(ruta_dir):
        os.makedirs(ruta_dir, exist_ok=True)

def ruta_resultado(nombre_archivo, base_dir):
    # devuelve la ruta absoluta dentro de la carpeta de resultados.
    return os.path.join(base_dir, nombre_archivo)

def fijar_semillas(semilla=42):
    # fija semillas para intentar reproducibilidad (no es perfecta, pero ayuda).
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

def guardar_historial(hist, ruta_png, titulo):
    # grafica y guarda las curvas de pérdida de entrenamiento y validación.
    # esto permite verificar si el modelo está aprendiendo o sobreajustando.
    plt.figure(figsize=(6,4))
    plt.plot(hist.history.get("loss", []), label="train")
    if "val_loss" in hist.history:
        plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("época")
    plt.ylabel("loss")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()

def cuadricula_imagenes(filas, cols, imagenes, ruta_png, titulo, reshape=None, cmap="gray"):
    # arma una grilla de imágenes (por filas y columnas) y la guarda como png.
    # 'reshape' permite re-formar vectores (784,) a imágenes (28,28) para visualizar.
    total = filas * cols
    imgs = imagenes[:total]
    plt.figure(figsize=(cols*1.4, filas*1.4))
    for i in range(total):
        plt.subplot(filas, cols, i + 1)
        img = imgs[i]
        if reshape is not None:
            img = img.reshape(reshape)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
    plt.suptitle(titulo, y=0.98, fontsize=12)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()

def guardar_texto(texto, ruta_txt):
    # guarda texto plano (por ejemplo, resúmenes de modelos o informes).
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(texto)

# ------------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS
# ------------------------------------------------------------

def cargar_mnist_normalizado():
    # carga mnist (imágenes 28x28 en escala de grises) desde keras.
    # normaliza a [0,1] y devuelve:
    #  - versión imagen (28x28) para plots
    #  - versión plana (784,) para alimentar la red densa
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train_img = x_train.copy()
    x_test_img  = x_test.copy()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test  = x_test.reshape((x_test.shape[0], -1))
    return (x_train, x_train_img), (x_test, x_test_img)

# ------------------------------------------------------------
# DEFINICIÓN DE MODELOS
# ------------------------------------------------------------

def construir_autoencoder_basico(input_dim=784, bottleneck=64):
    # autoencoder denso simétrico para reconstrucción:
    # encoder: input_dim -> 128 -> bottleneck
    # decoder: bottleneck -> 128 -> input_dim (sigmoid para salida en [0,1])
    entrada = layers.Input(shape=(input_dim,), name="entrada")
    x = layers.Dense(128, activation="relu")(entrada)
    latente = layers.Dense(bottleneck, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(latente)
    salida = layers.Dense(input_dim, activation="sigmoid")(x)

    modelo = models.Model(inputs=entrada, outputs=salida, name="autoencoder_basico")

    # optimizador adam con clipnorm: recorta la norma del gradiente para
    # estabilizar el entrenamiento (evita explosiones de pérdida).
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    modelo.compile(optimizer=opt, loss="binary_crossentropy")
    return modelo

def construir_autoencoder_denoising(input_dim=784, bottleneck=64):
    # autoencoder para denoising: misma arquitectura, pero entrenado con
    # (entrada ruidosa) -> (salida limpia). aprende a "limpiar" ruido gaussiano.
    entrada = layers.Input(shape=(input_dim,), name="entrada_ruidosa")
    x = layers.Dense(128, activation="relu")(entrada)
    latente = layers.Dense(bottleneck, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(latente)
    salida = layers.Dense(input_dim, activation="sigmoid")(x)

    modelo = models.Model(inputs=entrada, outputs=salida, name="autoencoder_denoising")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    modelo.compile(optimizer=opt, loss="binary_crossentropy")
    return modelo

# ------------------------------------------------------------
# GENERACIÓN DE RUIDO
# ------------------------------------------------------------

def agregar_ruido_gaussiano(x, media=0.0, sigma=0.5):
    # agrega ruido gaussiano a los datos y lo recorta a [0,1].
    # x puede ser (n, 784). este paso simula imágenes "sucias".
    ruido = np.random.normal(media, sigma, x.shape).astype("float32")
    x_ruidoso = x + ruido
    return np.clip(x_ruidoso, 0.0, 1.0)

# ------------------------------------------------------------
# ENTRENAMIENTO DE MODELOS
# ------------------------------------------------------------

def entrenar_autoencoder_basico(x_train, x_test, resultados_dir, epocas=15, batch=256):
    # entrena el autoencoder básico para reconstruir imágenes limpias.
    # entradas y salidas son las mismas (x -> x).
    modelo = construir_autoencoder_basico(input_dim=x_train.shape[1], bottleneck=64)

    # guardo el resumen para documentación del experimento.
    resumen = []
    modelo.summary(print_fn=lambda s: resumen.append(s))
    guardar_texto("\n".join(resumen), ruta_resultado("modelo_basico_resumen.txt", resultados_dir))

    # early stopping: si la val_loss deja de mejorar algunas épocas, se detiene y
    # restaura los mejores pesos. evita sobreajuste y ahorra tiempo.
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    hist = modelo.fit(
        x_train, x_train,
        validation_data=(x_test, x_test),
        epochs=epocas,
        batch_size=batch,
        callbacks=[es],
        verbose=2
    )

    # guardo las curvas de entrenamiento y unas reconstrucciones de muestra.
    guardar_historial(hist, ruta_resultado("curvas_loss_basico.png", resultados_dir), "autoencoder básico")
    x_pred = modelo.predict(x_test[:10], verbose=0)
    return modelo, x_pred, hist

def entrenar_autoencoder_denoising(x_train, x_test, resultados_dir, epocas=15, batch=256):
    # entrena el autoencoder de denoising.
    # x_train y x_test son tuplas: (x_ruidosa, x_limpia).
    modelo = construir_autoencoder_denoising(input_dim=x_train[0].shape[1], bottleneck=64)

    resumen = []
    modelo.summary(print_fn=lambda s: resumen.append(s))
    guardar_texto("\n".join(resumen), ruta_resultado("modelo_denoising_resumen.txt", resultados_dir))

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    hist = modelo.fit(
        x_train[0], x_train[1],           # entrada ruidosa -> salida limpia
        validation_data=(x_test[0], x_test[1]),
        epochs=epocas,
        batch_size=batch,
        callbacks=[es],
        verbose=2
    )

    guardar_historial(hist, ruta_resultado("curvas_loss_denoising.png", resultados_dir), "autoencoder denoising")
    x_pred = modelo.predict(x_test[0][:10], verbose=0)
    return modelo, x_pred, hist

# ------------------------------------------------------------
# EVALUACIÓN Y MÉTRICAS
# ------------------------------------------------------------

def evaluar_modelos(basico, denoise, x_test, resultados_dir):
    # calcula pérdidas de reconstrucción en test para ambos modelos:
    #  - básico: compara salida(básico(x_test)) vs x_test
    #  - denoising: evalúa su capacidad de limpiar ruido: denoise(noisy(x_test)) vs x_test
    bce = tf.keras.losses.BinaryCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()

    # básico (entrada limpia -> salida reconstruida)
    yb = basico.predict(x_test, verbose=0)
    loss_bce_basico = float(bce(x_test, yb).numpy())
    loss_mse_basico = float(mse(x_test, yb).numpy())

    # denoising (entrada ruidosa -> salida estimada) comparada con la limpia real.
    x_test_noisy = agregar_ruido_gaussiano(x_test, sigma=0.5)
    yd = denoise.predict(x_test_noisy, verbose=0)
    loss_bce_denoise = float(bce(x_test, yd).numpy())
    loss_mse_denoise = float(mse(x_test, yd).numpy())

    # guardo un resumen en json para uso posterior (o revisión rápida).
    resumen = {
        "loss_bce_basico": loss_bce_basico,
        "loss_mse_basico": loss_mse_basico,
        "loss_bce_denoise": loss_bce_denoise,
        "loss_mse_denoise": loss_mse_denoise
    }
    guardar_texto(json.dumps(resumen, indent=2, ensure_ascii=False),
                  ruta_resultado("metricas.json", resultados_dir))
    return resumen

# ------------------------------------------------------------
# VISUALIZACIONES PRINCIPALES
# ------------------------------------------------------------

def graficos_reconstruccion_basico(x_test_img, x_pred_flat, resultados_dir):
    # muestra dos filas (10 columnas):
    #   fila 1: originales
    #   fila 2: reconstruidas por el autoencoder básico
    originales = x_test_img[:10]
    reconstruidas = x_pred_flat.reshape((-1, 28, 28))
    imgs = np.vstack([originales, reconstruidas])
    cuadricula_imagenes(2, 10, imgs,
                        ruta_resultado("reconstrucciones_basico.png", resultados_dir),
                        "básico — fila1: original, fila2: reconstruida",
                        reshape=(28,28))

def graficos_denoising_completo(x_test_img, x_test_noisy_flat, x_pred_flat, resultados_dir):
    # muestra tres filas (10 columnas):
    #   fila 1: imágenes con ruido
    #   fila 2: salidas del modelo (denoised)
    #   fila 3: originales limpias
    ruidosas = x_test_noisy_flat.reshape((-1, 28, 28))[:10]
    denoised = x_pred_flat.reshape((-1, 28, 28))[:10]
    originales = x_test_img[:10]
    imgs = np.vstack([ruidosas, denoised, originales])
    cuadricula_imagenes(3, 10, imgs,
                        ruta_resultado("denoising_tripleta.png", resultados_dir),
                        "fila1: ruidosa, fila2: salida, fila3: original",
                        reshape=(28,28))

# ------------------------------------------------------------
# PUNTO DE ENTRADA (MAIN)
# ------------------------------------------------------------

def main():
    # argumentos por línea de comando para ajustar experimento sin tocar el código.
    parser = argparse.ArgumentParser(description="actividad sesión 3 — autoencoders")
    parser.add_argument("--epocas", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--sigma_ruido", type=float, default=0.5)
    args = parser.parse_args()

    # preparo carpeta de resultados junto al script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resultados_dir = os.path.join(base_dir, "resultados_sesion3")
    asegurar_directorio(resultados_dir)
    fijar_semillas(42)

    # cargo datos en dos formatos: imagen (28x28) y plano (784,).
    (x_train, x_train_img), (x_test, x_test_img) = cargar_mnist_normalizado()

    # -------- autoencoder básico --------
    t0 = time.time()
    modelo_basico, x_pred_basico, hist_b = entrenar_autoencoder_basico(
        x_train, x_test, resultados_dir,
        epocas=args.epocas, batch=args.batch
    )
    graficos_reconstruccion_basico(x_test_img, x_pred_basico, resultados_dir)
    t1 = time.time()

    # -------- autoencoder denoising --------
    # genero datasets ruidosos para entrenamiento y validación.
    x_train_noisy = agregar_ruido_gaussiano(x_train, sigma=args.sigma_ruido)
    x_test_noisy  = agregar_ruido_gaussiano(x_test, sigma=args.sigma_ruido)

    modelo_denoise, x_pred_denoise, hist_d = entrenar_autoencoder_denoising(
        (x_train_noisy, x_train), (x_test_noisy, x_test),
        resultados_dir,
        epocas=args.epocas, batch=args.batch
    )
    graficos_denoising_completo(x_test_img, x_test_noisy, x_pred_denoise, resultados_dir)
    t2 = time.time()

    # -------- métricas e informe breve --------
    metricas = evaluar_modelos(modelo_basico, modelo_denoise, x_test, resultados_dir)

    informe = []
    informe.append("actividad sesión 3 — autoencoders\n")
    informe.append(f"- básico entrenado en {t1 - t0:.1f}s")
    informe.append(f"- denoising entrenado en {t2 - t1:.1f}s\n")
    informe.append(f"loss test básico bce={metricas['loss_bce_basico']:.6f}, mse={metricas['loss_mse_basico']:.6f}")
    informe.append(f"loss test denoise bce={metricas['loss_bce_denoise']:.6f}, mse={metricas['loss_mse_denoise']:.6f}")
    guardar_texto("\n".join(informe), ruta_resultado("informe_resultados.txt", resultados_dir))

    # guardo los modelos entrenados para uso posterior (inspección o inferencia).
    modelo_basico.save(os.path.join(resultados_dir, "modelo_basico.keras"))
    modelo_denoise.save(os.path.join(resultados_dir, "modelo_denoising.keras"))

    print("\nlisto. resultados en:", resultados_dir)

if __name__ == "__main__":
    main()