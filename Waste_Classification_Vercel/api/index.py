import io
import json
import os

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.onnx")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
IMG_SIZE = 128

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # Vercel body limit is ~4.5 MB

_session = None
_class_names = None


def load_model():
    """Load lazily so cold starts stay fast and errors are readable."""
    global _session, _class_names
    if _session is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH)):
            raise FileNotFoundError(MODEL_PATH)
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            _class_names = json.load(f)
    return _session, _class_names


def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_waste(file_bytes):
    session, class_names = load_model()
    x = preprocess(file_bytes)
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: x})[0][0]
    index = int(np.argmax(probs))
    return {
        "prediction": class_names[index],
        "confidence": round(float(probs[index]) * 100, 2),
        "all": {class_names[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)},
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="Please select an image."), 400
    try:
        return jsonify(predict_waste(file.read()))
    except FileNotFoundError:
        return jsonify(error="Model files not found. Add models/waste_classifier.onnx and models/class_names.json."), 500
    except Exception as e:
        return jsonify(error=f"Prediction error: {e}"), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image too large (max 4 MB)."), 413


if __name__ == "__main__":
    app.run(debug=True)
