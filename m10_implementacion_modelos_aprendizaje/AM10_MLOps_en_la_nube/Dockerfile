
# imagen ligera de python
FROM python:3.11-slim

# variables para no generar __pycache__ innecesarios y logs ruidosos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# directorio de trabajo
WORKDIR /app

# instalo dependencias del sistema que ayudan a compilar scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# copio requirements e instalo
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copio el resto del código
COPY . /app

# entreno el modelo en tiempo de build si no existe (opcional pero útil)
# se puede comentar si prefieres entrenar antes
RUN python modelo.py

# expongo el puerto
EXPOSE 8000

# comando por defecto: gunicorn (2 workers es razonable para demo)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
