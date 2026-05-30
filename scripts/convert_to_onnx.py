"""
One-time script: Download model_weights.h5 from CricketShotClassification repo
and convert it to ONNX format for GPU inference via onnxruntime-gpu.

Requirements (only needed for this conversion step):
    pip install tensorflow tf2onnx requests

After running this script, TensorFlow is no longer needed at runtime.
"""

import os
import sys
import requests
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
REPO_H5_URL = (
    "https://github.com/RITIK-12/CricketShotClassification"
    "/raw/main/model_weights.h5"
)

H5_PATH   = r"C:\Cricket-Angle\models\shot_classifier\model_weights.h5"
ONNX_PATH = r"C:\Cricket-Angle\models\shot_classifier\shot_classifier.onnx"

os.makedirs(os.path.dirname(H5_PATH), exist_ok=True)


# ── 1. Download weights ────────────────────────────────────────────────────
def download_weights():
    if os.path.exists(H5_PATH):
        print(f"✅ H5 already exists: {H5_PATH}")
        return

    print(f"⬇  Downloading model_weights.h5 …")
    resp = requests.get(REPO_H5_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(H5_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded//(1<<20)} MB)", end="", flush=True)

    print(f"\n✅ Saved → {H5_PATH}  ({os.path.getsize(H5_PATH)//(1<<20)} MB)")


# ── 2. Rebuild model architecture (must match training code exactly) ───────
def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,           # weights loaded from h5 below
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.GRU(256, return_sequences=True),
        layers.GRU(128),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.load_weights(H5_PATH)
    print("✅ Weights loaded into model")
    return model


# ── 3. Convert to ONNX ────────────────────────────────────────────────────
def convert(model):
    import tensorflow as tf
    import tf2onnx
    import shutil

    if os.path.exists(ONNX_PATH):
        print(f"✅ ONNX already exists: {ONNX_PATH}")
        return

    print("🔄 Converting to ONNX via SavedModel …")

    # tf2onnx.from_keras breaks with Keras 3 (TF >= 2.16).
    # Save as SavedModel first — from_saved_model works with all TF versions.
    saved_dir = os.path.join(os.path.dirname(ONNX_PATH), "_tmp_saved_model")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 30, 224, 224, 3), dtype=tf.float32, name="input")
    ])
    def serving_fn(x):
        return model(x, training=False)

    tf.saved_model.save(
        model,
        saved_dir,
        signatures={"serving_default": serving_fn},
    )
    print(f"  SavedModel written → {saved_dir}")

    # Use CLI — more stable than Python API across tf2onnx versions
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", saved_dir,
            "--output",      ONNX_PATH,
            "--opset",       "13",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)

    shutil.rmtree(saved_dir, ignore_errors=True)

    size_mb = os.path.getsize(ONNX_PATH) / (1 << 20)
    print(f"✅ ONNX saved → {ONNX_PATH}  ({size_mb:.1f} MB)")


# ── 4. Quick sanity check with onnxruntime ────────────────────────────────
def verify():
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠  onnxruntime not installed — skipping verify")
        return

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )

    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    dummy = np.zeros((1, 30, 224, 224, 3), dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})

    print(f"✅ ONNX verify OK — output shape: {out[0].shape}  provider: {sess.get_providers()[0]}")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_weights()

    try:
        model = build_model()
    except Exception as e:
        print(f"\n❌ Model build failed: {e}")
        print("   Make sure TensorFlow is installed:  pip install tensorflow==2.13.0 tf2onnx")
        sys.exit(1)

    convert(model)
    verify()

    print("\n🎉 Done!  You can now use shot_classifier.onnx with onnxruntime-gpu.")
    print(f"   ONNX model path: {ONNX_PATH}")
