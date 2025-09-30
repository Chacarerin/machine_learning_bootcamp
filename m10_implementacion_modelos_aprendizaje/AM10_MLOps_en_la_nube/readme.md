# actividad módulo 10 — mlops en la nube (demo de código)

este readme es un marcador temporal. más abajo dejo comandos rápidos para correr ahora mismo.

## entrenamiento
```bash
python modelo.py
```

## correr api local
```bash
python app.py
# o con gunicorn (recomendado)
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

## docker
```bash
docker build -t modulo10_api:latest .
docker run --rm -p 8000:8000 modulo10_api:latest
```

## probar
```bash
curl -s http://localhost:8000/ | jq
curl -s -X POST http://localhost:8000/predict       -H "content-type: application/json"       -d @- <<'JSON'
{ "data": { "mean radius": 14.2, "mean texture": 20.1, "mean perimeter": 92.0,
            "mean area": 600.0, "mean smoothness": 0.1, "mean compactness": 0.1,
            "mean concavity": 0.05, "mean concave points": 0.05, "mean symmetry": 0.2,
            "mean fractal dimension": 0.06, "radius error": 0.3, "texture error": 1.0,
            "perimeter error": 2.0, "area error": 15.0, "smoothness error": 0.01,
            "compactness error": 0.02, "concavity error": 0.02, "concave points error": 0.01,
            "symmetry error": 0.02, "fractal dimension error": 0.003, "worst radius": 15.0,
            "worst texture": 22.0, "worst perimeter": 100.0, "worst area": 700.0,
            "worst smoothness": 0.12, "worst compactness": 0.15, "worst concavity": 0.1,
            "worst concave points": 0.08, "worst symmetry": 0.25, "worst fractal dimension": 0.07 } }
JSON
```
