from ultralytics import YOLO
import math

class YoloBallDetector:
    def __init__(self, model_path, conf=0.01, ball_class_id=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.ball_class_id = ball_class_id

        # 🔧 BALL SIZE FILTERS (tune later)
        self.min_area = 20
        self.max_area = 2500
        self.min_ratio = 0.6
        self.max_ratio = 1.4

    def detect(self, frame, predicted_pos=None):
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box, cls, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):
            if int(cls) != self.ball_class_id:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            area = w * h
            ratio = w / h

            # ❌ AREA FILTER
            if area < self.min_area or area > self.max_area:
                continue

            # ❌ ASPECT RATIO FILTER
            if ratio < self.min_ratio or ratio > self.max_ratio:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append(
                (cx, cy, x1, y1, x2, y2, float(conf))
            )

        return detections
