from ultralytics import YOLO
import torch
import numpy as np


class ShotClassifier:
    def __init__(self, model_path, conf=0.3, device=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf
        self.device = device if device is not None else (
            0 if torch.cuda.is_available() else "cpu"
        )

    def predict(self, frame):
        """
        Runs classification on full frame.
        Returns:
            (shot_label, confidence)
        """

        results = self.model.predict(
            source=frame,
            imgsz=224,
            device=self.device,
            verbose=False
        )

        if not results:
            return None, 0.0

        r = results[0]

        # 🔥 IMPORTANT: classification models use .probs
        if r.probs is None:
            return None, 0.0

        # Top-1 prediction
        top1_index = int(r.probs.top1)
        top1_conf = float(r.probs.top1conf)

        # Apply confidence threshold
        if top1_conf < self.conf_threshold:
            return None, top1_conf

        class_name = self.model.names[top1_index]

        return class_name, top1_conf
