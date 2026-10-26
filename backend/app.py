from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
TASK = BUNDLE.get("task", "classification")
LABELS = BUNDLE.get("labels")

# Websites that are allowed to call this API from a browser
ALLOWED_ORIGINS = [
    "https://bangash044.github.io",   # GitHub Pages site (origin has no path)
    "http://localhost:4200",          # local Angular
    "http://127.0.0.1:4200",
]

app = Flask(__name__)
CORS(app, resources={r"/health": {"origins": ALLOWED_ORIGINS},
                     r"/predict": {"origins": ALLOWED_ORIGINS}},
     methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"],
     max_age=86400)


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch",
                       missing=missing, extra=extra), 400
    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        prediction = MODEL.predict(frame)[0]
        prediction = (prediction.item() if hasattr(prediction, "item")
                      else prediction)
        result = {"prediction": prediction, "task": TASK,
                  "disclaimer": "Research/teaching prototype only."}
        if LABELS and isinstance(prediction, int):
            result["label"] = LABELS[prediction]
        if TASK == "classification" and hasattr(MODEL, "predict_proba"):
            result["probabilities"] = MODEL.predict_proba(frame)[0].tolist()
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)