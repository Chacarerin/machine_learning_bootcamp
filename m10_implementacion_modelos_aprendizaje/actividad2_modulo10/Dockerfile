# Dockerfile
FROM python:3.11-slim

# configuracion basica
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# crear directorio de trabajo
WORKDIR /app

# instalar dependencias del sistema si fueran necesarias (comentado; descomentar si hace falta)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# copiar requirements e instalar
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copiar codigo y artefactos (incluye resultados_sesion2/modelo.pkl)
COPY . /app

# exponer puerto
EXPOSE 5000

# variable de entorno con ruta del modelo (opcional)
ENV MODEL_PATH="/app/resultados_sesion2/modelo.pkl"

# comando por defecto: gunicorn con 2 workers
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
