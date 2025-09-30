
# pruebas mínimas de endpoints usando el test client de flask
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import app, FEATURES_ESPERADAS

def test_root_ok():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert isinstance(data["features"], list)

def test_predict_minimo():
    client = app.test_client()
    # armo un registro con ceros para probar validación del endpoint
    payload = {"data": {c: 0.0 for c in FEATURES_ESPERADAS}}
    r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["n"] == 1
    assert "resultados" in data
