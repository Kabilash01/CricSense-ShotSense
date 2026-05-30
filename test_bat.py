import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Cricket-Angle\models\bat_detector_v8n\weights\best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(
    r"C:\Cricket-Angle\videoplayback.mp4"
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.30,
        verbose=False
    )

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf = float(box.conf[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"BAT {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    cv2.imshow("BAT TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()