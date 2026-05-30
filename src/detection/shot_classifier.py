"""
GPU-accelerated cricket shot classifier using ONNX Runtime.

Model: EfficientNetB0 + GRU (converted from CricketShotClassification repo)
Input:  30 BGR video frames → resized to 224x224 → shape (1, 30, 224, 224, 3)
Output: shot class name + confidence %

Conversion (one-time):
    python scripts/convert_to_onnx.py

Runtime (no TensorFlow needed):
    pip install onnxruntime-gpu
"""

import cv2
import numpy as np
import onnxruntime as ort


SHOT_CLASSES = [
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep",
]

DISPLAY_NAMES = {
    "cover":      "Cover Drive",
    "defense":    "Defensive Shot",
    "flick":      "Flick",
    "hook":       "Hook Shot",
    "late_cut":   "Late Cut",
    "lofted":     "Lofted Drive",
    "pull":       "Pull Shot",
    "square_cut": "Square Cut",
    "straight":   "Straight Drive",
    "sweep":      "Sweep",
}

INPUT_SIZE = (224, 224)
N_FRAMES   = 30


class ShotClassifier:

    def __init__(self, onnx_path: str):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session    = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._provider  = self.session.get_providers()[0]
        print(f"[ShotClassifier] loaded on {self._provider}")

    def classify(self, frames: list) -> tuple:
        """
        Args:
            frames: list of BGR numpy arrays (any resolution).
                    Exactly N_FRAMES are used; fewer → pad last frame;
                    more → evenly sampled.
        Returns:
            (display_name: str, confidence_pct: float)
        """
        frames  = self._sample_frames(frames, N_FRAMES)
        tensor  = self._preprocess(frames)
        probs   = self.session.run(None, {self.input_name: tensor})[0][0]
        idx     = int(np.argmax(probs))
        label   = SHOT_CLASSES[idx]
        conf    = float(probs[idx]) * 100.0
        return DISPLAY_NAMES[label], round(conf, 2)

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_frames(frames: list, n: int) -> list:
        if not frames:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * n
        if len(frames) >= n:
            indices = np.linspace(0, len(frames) - 1, n, dtype=int)
            return [frames[i] for i in indices]
        pad = [frames[-1]] * (n - len(frames))
        return list(frames) + pad

    @staticmethod
    def _preprocess(frames: list) -> np.ndarray:
        processed = []
        for f in frames:
            resized = cv2.resize(f, INPUT_SIZE)
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            processed.append(rgb)
        arr = np.array(processed, dtype=np.float32)   # (30, 224, 224, 3)
        return np.expand_dims(arr, axis=0)             # (1, 30, 224, 224, 3)
