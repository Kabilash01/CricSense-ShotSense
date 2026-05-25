import cv2
import time
import json

from src.detection.yolo_detector import YoloBallDetector

from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

from src.events.ball_bat import BallBatContact

from src.analytics.trajectory_predictor import TrajectoryPredictor


# =========================================================
# CONFIG
# =========================================================

BALL_MODEL_PATH = r"C:\Cricket-Angle\ball_test\weights\best.pt"

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
# INIT
# =========================================================

ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.25,
    ball_class_id=0
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

future_trajectory = []
shot_angle = None
shot_name = None
predicted_distance = 0
contact_point = None

ball_analysis = []
events = []

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
    # BALL-BAT CONTACT
    # -----------------------------------------------------

    if tracker.initialized:

        vx, vy = tracker.get_velocity()

        contact, conf = contact_detector.detect(vx, vy)

        if contact:

            x, y = tracker.get_position()

            contact_point = (x, y)

            # ---------------------------------------------
            # FUTURE TRAJECTORY
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

                # angle-based shot name
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

                "timestamp_sec": round(
                    frame_idx / fps,
                    2
                ),

                "confidence": round(conf, 2),

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

        scale = perspective_scale(
            y,
            frame.shape[0]
        )

        speed = tracker.get_speed_kmph(
            METERS_PER_PIXEL,
            scale
        )

        cv2.putText(
            frame,
            f"Speed: {speed:.1f} km/h",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Max: {tracker.max_speed:.1f} km/h",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if tracker.detect_bounce():

            tracker.pitch_type = (
                tracker.classify_pitch(
                    frame.shape[0]
                )
            )

        if tracker.pitch_type:

            cv2.putText(
                frame,
                tracker.pitch_type,
                (20, 110),
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

            "max_speed_kmph": round(
                tracker.max_speed,
                2
            ),

            "pitch_type": tracker.pitch_type,

            "bounce_y_px": tracker.bounce_y,

            "shot_angle": shot_angle,

            "shot_name": shot_name,

            "predicted_distance_m": predicted_distance
        })

        tracker.reset()

        contact_detector.reset()

        future_trajectory = []
        shot_angle = None
        shot_name = None
        predicted_distance = 0
        contact_point = None

    # -----------------------------------------------------
    # FPS
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {display_fps}",
        (20, 160),
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