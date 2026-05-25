import cv2
from ultralytics import YOLO


MODEL_PATH = r"ball_test\weights\best.pt"
VIDEO_PATH = r"output_ball_tracking.mp4"


def main(model_path=MODEL_PATH, video_path=VIDEO_PATH):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Video not opened: {video_path}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.1, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"cls:{cls} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("YOLO ONLY TEST", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
