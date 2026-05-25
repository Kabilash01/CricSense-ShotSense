import cv2
import time
import json

from src.detection.yolo_detector import YoloBallDetector
from src.detection.boundary_detector import BoundaryLineDetector
from src.detection.shot_classifier import ShotClassifier   # 🔥 NEW
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball
from src.events.boundary_logic import intersects
from src.events.boundary_event import classify_boundary
from src.events.ball_bat import BallBatContact


# ---------------- CONFIG ----------------
BALL_MODEL_PATH = r"C:\CricketSense-Ball\ball_test\weights\best.pt"
BOUNDARY_MODEL_PATH = r"C:\CrickeSense-train\Boundary\runs\detect\boundary_detect\weights\best.pt"
SHOT_MODEL_PATH = r"C:\CrickeSense-train\Shot\runs\classify\yolov8m_shot_cls\weights\best.pt"   # 🔥 SET THIS
VIDEO_PATH = r"C:\Cricket-Angle\videoplayback.mp4"

OUTPUT_JSON = "ball_analysis.json"
EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520


def perspective_scale(y, h):
    return 1.0 + 0.6 * (y / h)


# ---------------- INIT ----------------
ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.25,
    ball_class_id=0
)

boundary_detector = BoundaryLineDetector(
    model_path=BOUNDARY_MODEL_PATH,
    conf=0.35
)

shot_classifier = ShotClassifier(   # 🔥 INIT ONCE
    model_path=SHOT_MODEL_PATH,
    conf=0.3
)

tracker = BallTracker()
contact_detector = BallBatContact()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS) or 30


# ---------------- STATE ----------------
ball_id = 0
boundary_fired = False

deliveries = []
events = []

# ---------------- SCOREBOARD ----------------
total_runs = 0
balls = 0
fours = 0
sixes = 0

# FPS
fps_frames, fps_time, display_fps = 0, time.time(), 0
frame_idx = 0


# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # ---------- FPS ----------
    fps_frames += 1
    if time.time() - fps_time >= 1.0:
        display_fps = fps_frames
        fps_frames = 0
        fps_time = time.time()

    # ---------- BALL DETECTION ----------
    predicted = tracker.predict() if tracker.initialized else None
    ball_detections = ball_detector.detect(frame, predicted)

    # ---------- BOUNDARY DETECTION ----------
    boundary_boxes = boundary_detector.detect(frame)

    # ---------- DRAW BOUNDARY ----------
    for bx1, by1, bx2, by2, _ in boundary_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 0), 2)

    # ---------- DRAW BALL ----------
    for cx, cy, x1, y1, x2, y2, _ in ball_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # ---------- TRACKER UPDATE ----------
    if ball_detections:
        tracker.missed_frames = 0

        if not tracker.initialized:
            ball_id += 1
            cx, cy, *_ = ball_detections[0]
            tracker.update((cx, cy))
        else:
            match = associate_ball(ball_detections, predicted)
            if match:
                cx, cy, *_ = match
                tracker.update((cx, cy))
    else:
        tracker.missed_frames += 1

    # ---------------- BALL–BAT CONTACT ----------------
    if tracker.initialized:
        vx, vy = tracker.get_velocity()
        contact, conf = contact_detector.detect(vx, vy)

        if contact:
            # 🔥 RUN SHOT MODEL ONLY HERE
            shot_label, shot_conf = shot_classifier.predict(frame)

            event = {
                "ball_id": ball_id,
                "event": "ball_bat_contact",
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 2),
                "confidence": round(conf, 2),
                "shot_type": shot_label,
                "shot_confidence": round(shot_conf, 2)
            }

            events.append(event)
            print("🏏 BAT CONTACT:", event)

            x, y = tracker.get_position()

            # Visual marker
            cv2.circle(frame, (x, y), 12, (255, 0, 0), 3)

            if shot_label:
                cv2.putText(
                    frame,
                    shot_label,
                    (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )

    # ---------- BOUNDARY EVENT ----------
    if tracker.initialized and not boundary_fired:
        x, y = tracker.get_position()
        ball_box = (x - 6, y - 6, x + 6, y + 6)

        for bx1, by1, bx2, by2, _ in boundary_boxes:
            boundary_box = (bx1, by1, bx2, by2)

            if intersects(ball_box, boundary_box):
                boundary_type = classify_boundary(
                    tracker.get_speed_kmph(METERS_PER_PIXEL),
                    tracker.has_bounced
                )

                if boundary_type == "FOUR":
                    total_runs += 4
                    fours += 1
                elif boundary_type == "SIX":
                    total_runs += 6
                    sixes += 1

                event = {
                    "ball_id": ball_id,
                    "event": "boundary",
                    "type": boundary_type,
                    "frame": frame_idx,
                    "timestamp_sec": round(
                        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2
                    )
                }

                events.append(event)
                boundary_fired = True

                print("🏏 BOUNDARY:", event)

                cv2.putText(
                    frame,
                    boundary_type,
                    (frame.shape[1] // 2 - 120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 0, 255),
                    5
                )
                break

    # ---------- DELIVERY END ----------
    if tracker.missed_frames > 15 and tracker.initialized:
        deliveries.append({
            "ball_id": ball_id,
            "max_speed_kmph": round(tracker.max_speed, 2),
            "pitch_type": tracker.pitch_type,
            "bounce_y_px": tracker.bounce_y
        })

        balls += 1
        tracker.reset()
        contact_detector.reset()
        boundary_fired = False

    # ---------- SCOREBOARD ----------
    overs = balls // 6
    balls_in_over = balls % 6

    score_text = f"{total_runs}/{balls}  ({overs}.{balls_in_over} ov)"
    stats_text = f"4s: {fours}   6s: {sixes}"

    cv2.rectangle(frame, (10, 10), (380, 95), (0, 0, 0), -1)

    cv2.putText(frame, score_text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    cv2.putText(frame, stats_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS: {display_fps}", (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("CricketSense | Broadcast View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------- SAVE JSON ----------------
cap.release()
cv2.destroyAllWindows()

with open(OUTPUT_JSON, "w") as f:
    json.dump(deliveries, f, indent=4)

with open(EVENTS_JSON, "w") as f:
    json.dump(events, f, indent=4)

print(f"✅ Saved {len(events)} events → {EVENTS_JSON}")
print(f"✅ Saved {len(deliveries)} deliveries → {OUTPUT_JSON}")
