from ultralytics import YOLO


class BatDetector:

    def __init__(self, model_path, conf=0.10, bat_class_id=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.bat_class_id = bat_class_id  # None = accept all classes
        self._debug_printed = False

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            verbose=False
        )

        detections = []

        for r in results:

            # Print class names once so we know what the model detects
            if not self._debug_printed and r.boxes is not None and len(r.boxes):
                print(f"[BatDetector] model classes: {r.names}")
                self._debug_printed = True

            if r.boxes is None:
                continue

            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                # Filter by class only if bat_class_id is specified
                if self.bat_class_id is not None and cls != self.bat_class_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                # Bat must be a minimum size (stumps/helmet are smaller)
                if h < 40 or w < 10:
                    continue

                # Bat is always taller than wide when held vertically
                # Relax to 0.8 to also catch slightly angled bats
                aspect = h / max(w, 1)
                if aspect < 0.8:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append((cx, cy, x1, y1, x2, y2, conf))

        return detections
