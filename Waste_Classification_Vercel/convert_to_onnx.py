"""Convert the trained Keras model to ONNX (run locally after train.py)."""
import tensorflow as tf

KERAS_PATH = "models/waste_classifier.keras"
ONNX_PATH = "models/waste_classifier.onnx"

model = tf.keras.models.load_model(KERAS_PATH)

try:
    # Keras 3.2+ (needs: pip install tf2onnx onnx)
    model.export(ONNX_PATH, format="onnx")
except Exception as e:
    print("model.export(format='onnx') failed:", e)
    print("Fallback: exporting SavedModel, then run:")
    model.export("models/saved_model")
    print("  python -m tf2onnx.convert --saved-model models/saved_model "
          f"--output {ONNX_PATH} --opset 13")
    raise SystemExit(1)

print("Saved:", ONNX_PATH)
