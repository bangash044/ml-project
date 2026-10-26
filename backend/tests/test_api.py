import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import app, FEATURES

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["features"] == len(FEATURES)


def test_predict_valid():
    response = app.test_client().post("/predict", json=PAYLOAD)
    assert response.status_code == 200, response.get_json()
    assert "prediction" in response.get_json()


def test_predict_missing_feature():
    incomplete = dict(PAYLOAD)
    incomplete.pop(FEATURES[0])
    response = app.test_client().post("/predict", json=incomplete)
    assert response.status_code == 400


def test_predict_invalid_json():
    response = app.test_client().post("/predict", data="not-json",
                                      content_type="application/json")
    assert response.status_code == 400


def test_cors_preflight_allows_pages_origin():
    response = app.test_client().options(
        "/predict",
        headers={"Origin": "https://bangash044.github.io",
                 "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"})
    assert response.status_code in (200, 204)
    assert response.headers.get("Access-Control-Allow-Origin") == "https://bangash044.github.io"


def test_cors_blocks_unknown_origin():
    response = app.test_client().options(
        "/predict",
        headers={"Origin": "https://evil.example",
                 "Access-Control-Request-Method": "POST"})
    assert "Access-Control-Allow-Origin" not in response.headers