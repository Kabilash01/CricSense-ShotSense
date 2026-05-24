import cv2
import time
import json
import numpy as np

from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\CricketSense-Ball\ball_test\weights\best.pt"
VIDEO_PATH = r"C:\CricketSense\data\samples\test3.mp4"

OUTPUT_JSON = "ball_analysis.json"
OUTPUT_VIDEO = "ball_tracking_output.mp4"

METERS_PER_PIXEL = 18.5 / 520

def perspective_scale(y, h):
    return 1.0 + 0.6 * (y / h)

# ---------------- HELPERS ----------------
def draw_trajectory_curve(frame, points):
    if len(points) < 6:
        return
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    try:
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        xs = np.linspace(x.min(), x.max(), 50)
        ys = poly(xs)
        for i in range(1, len(xs)):
            cv2.line(frame,
                     (int(xs[i-1]), int(ys[i-1])),
                     (int(xs[i]), int(ys[i])),
                     (255, 0, 0), 2)
    except:
        pass

def draw_pitch_map(frame, pitch_type):
    h, w, _ = frame.shape
    overlay = frame.copy()
    zones = {
        "YORKER": (0.75, 1.0, (0,0,255)),
        "FULL":   (0.55, 0.75, (0,255,255)),
        "GOOD":   (0.35, 0.55, (0,255,0)),
        "SHORT":  (0.0, 0.35, (255,0,0))
    }
    if pitch_type not in zones:
        return
    y1, y2, color = zones[pitch_type]
    cv2.rectangle(overlay,
                  (0, int(y1*h)),
                  (w, int(y2*h)),
                  color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

# ---------------- INIT ----------------
detector = YoloBallDetector(MODEL_PATH, conf=0.2, ball_class_id=0)
tracker = BallTracker()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened()

# ---------- VIDEO WRITER ----------
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps_in,
    (width, height)
)

deliveries = []
ball_id = 0

fps_frames = 0
fps_time = time.time()
display_fps = 0

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps_frames += 1
    if time.time() - fps_time >= 1.0:
        display_fps = fps_frames
        fps_frames = 0
        fps_time = time.time()

    predicted = tracker.predict() if tracker.initialized else None
    detections = detector.detect(frame, predicted)

    for det in detections:
        cx, cy, x1, y1, x2, y2, _ = det
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(frame,(cx,cy),3,(0,0,255),-1)

    if detections:
        tracker.missed_frames = 0
        if not tracker.initialized:
            cx, cy, *_ = detections[0]
            tracker.update((cx,cy))
        else:
            match = associate_ball(detections, predicted)
            if match:
                cx, cy, *_ = match
                tracker.update((cx,cy))
    else:
        tracker.missed_frames += 1

    if tracker.missed_frames > 15 and tracker.initialized:
        ball_id += 1
        deliveries.append({
            "ball_id": ball_id,
            "release_speed_kmph": round(tracker.release_speed or 0, 2),
            "max_speed_kmph": round(tracker.max_speed, 2),
            "pitch_type": tracker.pitch_type,
            "bounce_y_px": tracker.bounce_y,
            "release_point": tracker.release_point
        })
        tracker.reset()

    if tracker.initialized:
        x, y = tracker.get_position()
        cv2.circle(frame,(x,y),6,(0,0,255),-1)

        draw_trajectory_curve(frame, tracker.trajectory)

        if tracker.release_point:
            rx, ry = tracker.release_point
            cv2.circle(frame,(rx,ry),8,(255,0,0),-1)
            cv2.putText(frame,"RELEASE",(rx+8,ry-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        scale = perspective_scale(y, frame.shape[0])
        speed = tracker.get_speed_kmph(METERS_PER_PIXEL, scale)

        cv2.putText(frame,f"Speed: {speed:.1f} km/h",(20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        if tracker.release_speed:
            cv2.putText(frame,f"Release: {tracker.release_speed:.1f} km/h",
                        (20,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        cv2.putText(frame,f"Max: {tracker.max_speed:.1f} km/h",
                    (20,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(frame.shape[0])

        if tracker.pitch_type:
            draw_pitch_map(frame, tracker.pitch_type)
            cv2.putText(frame, tracker.pitch_type,(20,130),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)

    cv2.putText(frame,f"FPS: {display_fps}",(20,170),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    # ✅ WRITE FRAME TO OUTPUT VIDEO
    video_out.write(frame)

    cv2.imshow("Cricket Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
video_out.release()
cv2.destroyAllWindows()

with open(OUTPUT_JSON,"w") as f:
    json.dump(deliveries,f,indent=4)

print("✅ JSON saved:", OUTPUT_JSON)
print("🎥 Video saved:", OUTPUT_VIDEO)
