import requests

# endpoint del api flask
url = "http://192.168.1.83:8000/predict"

# payload con todas las features necesarias
payload = {
    "data": {
        "mean radius": 14.2,
        "mean texture": 20.1,
        "mean perimeter": 92.0,
        "mean area": 600.0,
        "mean smoothness": 0.1,
        "mean compactness": 0.1,
        "mean concavity": 0.05,
        "mean concave points": 0.05,
        "mean symmetry": 0.2,
        "mean fractal dimension": 0.06,
        "radius error": 0.3,
        "texture error": 1.0,
        "perimeter error": 2.0,
        "area error": 15.0,
        "smoothness error": 0.01,
        "compactness error": 0.02,
        "concavity error": 0.02,
        "concave points error": 0.01,
        "symmetry error": 0.02,
        "fractal dimension error": 0.003,
        "worst radius": 15.0,
        "worst texture": 22.0,
        "worst perimeter": 100.0,
        "worst area": 700.0,
        "worst smoothness": 0.12,
        "worst compactness": 0.15,
        "worst concavity": 0.1,
        "worst concave points": 0.08,
        "worst symmetry": 0.25,
        "worst fractal dimension": 0.07
    }
}

# enviar petición post
resp = requests.post(url, json=payload)

# mostrar resultado
print(resp.status_code)
print(resp.json())