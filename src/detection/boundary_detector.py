from ultralytics import YOLO

class BoundaryLineDetector:
    def __init__(self, model_path, conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Returns list of boundary bounding boxes
        Each bbox: (x1, y1, x2, y2, conf)
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        boxes = []

        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                cls = int(b.cls[0])
                if cls == 0:  # boundary line class
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))

        return boxes
