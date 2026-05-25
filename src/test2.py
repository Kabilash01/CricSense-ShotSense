# Updated `src/test2.py`

import cv2
import time
import json
from src.detection.yolo_detector import YoloBallDetector
from src.detection.boundary_detector import BoundaryLineDetector
from src.detection.shot_classifier import ShotClassifier
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball
from src.events.boundary_logic import intersects
from src.events.boundary_event import classify_boundary
from src.events.ball_bat import BallBatContact
from src.analytics.trajectory_predictor import TrajectoryPredictor


# =========================================================
# CONFIG
# =========================================================

BALL_MODEL_PATH = r"C:\CricketSense-Ball\ball_test\weights\best.pt"

BOUNDARY_MODEL_PATH = (
    r"C:\CrickeSense-train\Boundary\runs\detect"
    r"\boundary_detect\weights\best.pt"
)

SHOT_MODEL_PATH = (
    r"C:\CrickeSense-train\Shot\runs\classify"
    r"\yolov8m_shot_cls\weights\best.pt"
)

VIDEO_PATH = r"C:\Cricket-Angle\videoplayback.mp4"

OUTPUT_JSON = "ball_analysis.json"
EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520


# =========================================================
# HELPERS
# =========================================================

def perspective_scale(y, h):
    return 1.0 + 0.6 * (y / h)


# =========================================================
# INIT MODELS
# =========================================================

ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.25,
    ball_class_id=0
)

boundary_detector = BoundaryLineDetector(
    model_path=BOUNDARY_MODEL_PATH,
    conf=0.35
)

shot_classifier = ShotClassifier(
    model_path=SHOT_MODEL_PATH,
    conf=0.30
)

tracker = BallTracker()

contact_detector = BallBatContact()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS) or 30

trajectory_predictor = TrajectoryPredictor(
    fps=fps,
    friction=0.985,
    predict_seconds=2.0
)


# =========================================================
# GLOBAL STATE
# =========================================================

ball_id = 0
boundary_fired = False

# trajectory state
future_trajectory = []
shot_angle = None
shot_name = None
predicted_distance = 0
contact_point = None

# outputs
ball_analysis = []
events = []

# scoreboard
runs = 0
balls = 0
fours = 0
sixes = 0

# fps display
fps_frames = 0
fps_time = time.time()
display_fps = 0

frame_idx = 0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    # -----------------------------------------------------
    # FPS
    # -----------------------------------------------------

    fps_frames += 1

    if time.time() - fps_time >= 1.0:
        display_fps = fps_frames
        fps_frames = 0
        fps_time = time.time()

    # -----------------------------------------------------
    # BALL DETECTION
    # -----------------------------------------------------

    predicted = tracker.predict() if tracker.initialized else None

    ball_detections = ball_detector.detect(frame, predicted)

    # -----------------------------------------------------
    # BOUNDARY DETECTION
    # -----------------------------------------------------

    boundary_boxes = boundary_detector.detect(frame)

    # -----------------------------------------------------
    # DRAW BOUNDARY
    # -----------------------------------------------------

    for bx1, by1, bx2, by2, _ in boundary_boxes:

        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (255, 255, 0),
            2
        )

    # -----------------------------------------------------
    # DRAW BALL DETECTIONS
    # -----------------------------------------------------

    for cx, cy, x1, y1, x2, y2, _ in ball_detections:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            3,
            (0, 0, 255),
            -1
        )

    # -----------------------------------------------------
    # TRACKER UPDATE
    # -----------------------------------------------------

    if ball_detections:

        tracker.missed_frames = 0

        if not tracker.initialized:

            ball_id += 1

            cx, cy, *_ = ball_detections[0]

            tracker.update((cx, cy))

        else:

            match = associate_ball(
                ball_detections,
                predicted
            )

            if match:

                cx, cy, *_ = match

                tracker.update((cx, cy))

    else:

        tracker.missed_frames += 1

    # -----------------------------------------------------
    # BALL - BAT CONTACT
    # -----------------------------------------------------

    if tracker.initialized:

        vx, vy = tracker.get_velocity()

        contact, conf = contact_detector.detect(vx, vy)

        if contact:

            x, y = tracker.get_position()

            contact_point = (x, y)

            # ---------------------------------------------
            # SHOT MODEL
            # ---------------------------------------------

            shot_label, shot_conf = shot_classifier.predict(frame)

            # ---------------------------------------------
            # FUTURE TRAJECTORY PREDICTION
            # ---------------------------------------------

            future_trajectory = trajectory_predictor.predict(
                x=x,
                y=y,
                vx=vx,
                vy=vy
            )

            # ---------------------------------------------
            # SHOT ANGLE
            # ---------------------------------------------

            if len(future_trajectory) > 0:

                final_point = future_trajectory[-1]

                shot_angle = trajectory_predictor.compute_angle(
                    contact_point,
                    final_point
                )

                shot_name = trajectory_predictor.classify_shot(
                    shot_angle
                )

                predicted_distance = (
                    trajectory_predictor.trajectory_distance_m(
                        future_trajectory,
                        METERS_PER_PIXEL
                    )
                )

            # ---------------------------------------------
            # EVENT JSON
            # ---------------------------------------------

            event = {
                "ball_id": ball_id,
                "event": "ball_bat_contact",
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 2),
                "confidence": round(conf, 2),
                "shot_type_model": shot_label,
                "shot_model_confidence": round(shot_conf, 2),
                "wagon_wheel_angle": shot_angle,
                "shot_name": shot_name,
                "predicted_distance_m": predicted_distance,
                "contact_point": {
                    "x": x,
                    "y": y
                },
                "future_trajectory": future_trajectory[:40]
            }

            events.append(event)

            print("🏏 SHOT EVENT:", event)

            # ---------------------------------------------
            # VISUALS
            # ---------------------------------------------

            cv2.circle(
                frame,
                (x, y),
                12,
                (255, 0, 0),
                3
            )

            if shot_name:

                cv2.putText(
                    frame,
                    f"{shot_name} | {shot_angle:.1f} deg",
                    (x + 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    3
                )

    # -----------------------------------------------------
    # TRAJECTORY OVERLAY
    # -----------------------------------------------------

    if len(future_trajectory) > 1:

        for i in range(1, len(future_trajectory)):

            cv2.line(
                frame,
                future_trajectory[i - 1],
                future_trajectory[i],
                (255, 0, 255),
                2
            )

        if contact_point:

            cv2.line(
                frame,
                contact_point,
                future_trajectory[-1],
                (0, 255, 255),
                3
            )

    # -----------------------------------------------------
    # BOUNDARY EVENT
    # -----------------------------------------------------

    if tracker.initialized and not boundary_fired:

        x, y = tracker.get_position()

        ball_box = (
            x - 6,
            y - 6,
            x + 6,
            y + 6
        )

        for bx1, by1, bx2, by2, _ in boundary_boxes:

            boundary_box = (
                bx1,
                by1,
                bx2,
                by2
            )

            if intersects(ball_box, boundary_box):

                boundary_type = classify_boundary(
                    tracker.get_speed_kmph(
                        METERS_PER_PIXEL
                    ),
                    tracker.has_bounced
                )

                if boundary_type == "FOUR":
                    runs += 4
                    fours += 1

                elif boundary_type == "SIX":
                    runs += 6
                    sixes += 1

                event = {
                    "ball_id": ball_id,
                    "event": "boundary",
                    "type": boundary_type,
                    "frame": frame_idx,
                    "timestamp_sec": round(
                        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                        2
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

    # -----------------------------------------------------
    # TRACKER VISUALS
    # -----------------------------------------------------

    if tracker.initialized:

        x, y = tracker.get_position()

        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        scale = perspective_scale(y, frame.shape[0])

        speed = tracker.get_speed_kmph(
            METERS_PER_PIXEL,
            scale
        )

        cv2.putText(
            frame,
            f"Speed: {speed:.1f} km/h",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Max: {tracker.max_speed:.1f} km/h",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if tracker.detect_bounce():

            tracker.pitch_type = tracker.classify_pitch(
                frame.shape[0]
            )

        if tracker.pitch_type:

            cv2.putText(
                frame,
                tracker.pitch_type,
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

    # -----------------------------------------------------
    # DELIVERY END
    # -----------------------------------------------------

    if tracker.missed_frames > 15 and tracker.initialized:

        ball_analysis.append({
            "ball_id": ball_id,
            "max_speed_kmph": round(tracker.max_speed, 2),
            "pitch_type": tracker.pitch_type,
            "bounce_y_px": tracker.bounce_y,
            "shot_angle": shot_angle,
            "shot_name": shot_name,
            "predicted_distance_m": predicted_distance
        })

        balls += 1

        tracker.reset()
        contact_detector.reset()

        boundary_fired = False

        # reset trajectory state
        future_trajectory = []
        shot_angle = None
        shot_name = None
        predicted_distance = 0
        contact_point = None

    # -----------------------------------------------------
    # SCOREBOARD
    # -----------------------------------------------------

    overs = balls // 6
    balls_in_over = balls % 6

    score_text = f"{runs}/{balls} ({overs}.{balls_in_over} ov)"

    stats_text = f"4s: {fours}   6s: {sixes}"

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 95),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        score_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        stats_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # -----------------------------------------------------
    # FPS
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {display_fps}",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(
        "CricketSense | Trajectory Intelligence",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================================================
# SAVE JSON
# =========================================================

cap.release()
cv2.destroyAllWindows()

with open(OUTPUT_JSON, "w") as f:
    json.dump(ball_analysis, f, indent=4)

with open(EVENTS_JSON, "w") as f:
    json.dump(events, f, indent=4)

print(f"✅ Saved {len(events)} events → {EVENTS_JSON}")
print(f"✅ Saved {len(ball_analysis)} deliveries → {OUTPUT_JSON}")

