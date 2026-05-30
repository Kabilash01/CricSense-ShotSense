import cv2
import time
import json
import math
import numpy as np

from collections import deque

from src.detection.yolo_detector import YoloBallDetector
from src.detection.bat_detector import BatDetector
from src.detection.shot_classifier import ShotClassifier
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball


# =========================================================
# CONFIG
# =========================================================

BALL_MODEL_PATH = r"C:\Cricket-Angle\ball_test\weights\best.pt"
BAT_MODEL_PATH  = r"C:\Cricket-Angle\models\bat_detector_v8n\weights\best.pt"
SHOT_ONNX_PATH  = r"C:\Cricket-Angle\models\shot_classifier\shot_classifier.onnx"

VIDEO_PATH = r"C:\Cricket-Angle\videoplayback.mp4"

OUTPUT_VIDEO = "output_analysis.mp4"
EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520


# =========================================================
# INIT
# =========================================================

ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.10,
    ball_class_id=0
)

bat_detector = BatDetector(
    model_path=BAT_MODEL_PATH,
    conf=0.05,
    bat_class_id=1       # class 0='-', class 1='bat'
)

import os as _os
shot_classifier = (
    ShotClassifier(SHOT_ONNX_PATH)
    if _os.path.exists(SHOT_ONNX_PATH)
    else None
)
if shot_classifier is None:
    print("⚠  shot_classifier.onnx not found — run scripts/convert_to_onnx.py first")
    print("   Falling back to geometry-based shot classification")

tracker = BallTracker()

cap = cv2.VideoCapture(VIDEO_PATH)

assert cap.isOpened(), "❌ Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS) or 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =========================================================
# OUTPUT VIDEO
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# =========================================================
# GLOBALS
# =========================================================

events = []

ball_id = 0

frame_idx = 0

fps_frames = 0
fps_time = time.time()
display_fps = 0

# =========================================================
# TRAJECTORY
# =========================================================

trajectory_history = deque(maxlen=30)

post_contact_points = []

future_trajectory = []

# Speed history for acceleration detection
speed_history = deque(maxlen=15)

# Rolling 60-frame buffer of raw BGR frames for shot classifier
frame_buffer = deque(maxlen=60)

# Frames captured around contact for shot classification
contact_frames = []

# Latest bat bounding box (x1,y1,x2,y2) or None
bat_box = None

# =========================================================
# CONTACT STATE
# =========================================================

contact_detected = False
contact_frame = -999

# =========================================================
# SHOT INFO
# =========================================================

shot_angle = None
shot_name = None
predicted_distance = 0

# =========================================================
# HELPERS
# =========================================================

def calculate_speed(vx, vy):
    return math.sqrt(vx**2 + vy**2)


def ball_near_bat(ball_pos, bat_box, margin=30):
    """Return True when ball centre is inside (or within margin of) the bat box."""
    bx, by = ball_pos
    x1, y1, x2, y2 = bat_box
    return (x1 - margin <= bx <= x2 + margin and
            y1 - margin <= by <= y2 + margin)


def calculate_motion_angle(vx, vy):
    angle = math.degrees(
        math.atan2(-vy, vx)
    )
    if angle < 0:
        angle += 360
    return angle


def detect_trajectory_curvature(trajectory_points, window_size=5):
    """
    Detect trajectory curvature by comparing angles at different segments.
    High curvature → direction change → possible bat contact
    """
    if len(trajectory_points) < window_size + 2:
        return 0.0

    # Recent segment
    p1 = np.array(trajectory_points[-window_size - 1])
    p2 = np.array(trajectory_points[-1])
    recent_vector = p2 - p1

    # Older segment
    p3 = np.array(trajectory_points[-window_size * 2 - 1])
    p4 = np.array(trajectory_points[-window_size - 1])
    older_vector = p4 - p3

    # Normalize to magnitude for angle comparison
    mag_recent = np.linalg.norm(recent_vector)
    mag_older = np.linalg.norm(older_vector)

    if mag_recent < 1 or mag_older < 1:
        return 0.0

    # Angle between vectors
    cos_angle = np.dot(
        recent_vector / mag_recent,
        older_vector / mag_older
    )
    cos_angle = np.clip(cos_angle, -1, 1)
    curvature = math.degrees(math.acos(cos_angle))

    return curvature


def detect_acceleration_spike(speed_history, threshold=5.0, window=3):
    """
    Detect sudden speed changes (acceleration/deceleration).
    Contact with bat causes rapid speed change.
    """
    if len(speed_history) < window:
        return 0.0

    # Compare average speed before and after
    recent_avg = np.mean(speed_history[-window:])
    older_avg = np.mean(speed_history[-(window*2):-window])

    acceleration = abs(recent_avg - older_avg)
    return acceleration


def classify_shot(angle):
    # Normalize angle to 0-360 range
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360

    # Ranges ordered by priority
    if 70 <= angle < 110:
        return "Straight Drive"

    elif 110 <= angle < 150:
        return "Cover Drive"

    elif 150 <= angle < 210:
        return "Pull Shot"

    elif 210 <= angle < 250:
        return "Sweep"

    elif 250 <= angle < 320:
        return "Leg Glance"

    elif 320 <= angle <= 360 or 0 <= angle < 70:
        # Both Flick (320-360) and Square Drive (0-70)
        if 320 <= angle <= 360:
            return "Flick"
        else:
            return "Square Drive"

    else:
        return "Cut Shot"


def smooth_trajectory(points):

    if len(points) < 5:
        return points

    pts = np.array(points)

    x = pts[:, 0]
    y = pts[:, 1]

    kernel = np.ones(5) / 5

    smooth_x = np.convolve(
        x,
        kernel,
        mode='valid'
    )

    smooth_y = np.convolve(
        y,
        kernel,
        mode='valid'
    )

    return list(
        zip(
            smooth_x.astype(int),
            smooth_y.astype(int)
        )
    )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1
    frame_buffer.append(frame.copy())

    # =====================================================
    # FPS
    # =====================================================

    fps_frames += 1

    if time.time() - fps_time >= 1.0:

        display_fps = fps_frames

        fps_frames = 0

        fps_time = time.time()

    # =====================================================
    # DETECTION
    # =====================================================

    predicted = (
        tracker.predict()
        if tracker.initialized
        else None
    )

    detections = ball_detector.detect(
        frame,
        predicted
    )

    # =====================================================
    # BAT DETECTION
    # =====================================================

    bat_detections = bat_detector.detect(frame)

    if bat_detections:
        # Use the highest-confidence bat detection
        bat_detections.sort(key=lambda d: d[6], reverse=True)
        _, _, bx1, by1, bx2, by2, bat_conf = bat_detections[0]
        bat_box = (bx1, by1, bx2, by2)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 165, 0), 2)
        cv2.putText(
            frame, f"BAT {bat_conf:.2f}",
            (bx1, by1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2
        )

    # =====================================================
    # DRAW BALL DETECTIONS
    # =====================================================

    for cx, cy, x1, y1, x2, y2, conf in detections:

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
            4,
            (0, 0, 255),
            -1
        )

    # =====================================================
    # TRACKER UPDATE
    # =====================================================

    tracker_updated = False

    if detections:

        tracker.missed_frames = 0

        if not tracker.initialized:

            ball_id += 1

            cx, cy, *_ = detections[0]

            tracker.update((cx, cy))
            tracker_updated = True

        else:

            match = associate_ball(
                detections,
                predicted
            )

            if match:

                cx, cy, *_ = match

                tracker.update((cx, cy))
                tracker_updated = True

    else:

        tracker.missed_frames += 1

        # Reset tracker if ball has been missing too long
        if tracker.missed_frames > 30:
            tracker.reset()
            trajectory_history.clear()
            speed_history.clear()

    # =====================================================
    # TRACKER LOGIC
    # =====================================================

    if tracker.initialized:

        x, y = tracker.get_position()

        vx, vy = tracker.get_velocity()

        speed = calculate_speed(vx, vy)

        # Phantom speed guard: tracker jumped to a false detection far away
        # Reset if speed is physically impossible (> 800 px/frame)
        if speed > 800:
            tracker.reset()
            trajectory_history.clear()
            speed_history.clear()
            print(f"[F{frame_idx}] ⚠ Phantom speed {speed:.0f} — tracker reset")
            continue

        trajectory_history.append((x, y))
        speed_history.append(speed)

        # =================================================
        # CONTACT DETECTION
        # Priority 1: bat box (precise)
        # Priority 2: trajectory signals (always active)
        # =================================================

        cooldown_ok = not contact_detected and (frame_idx - contact_frame) > 40

        if cooldown_ok:

            triggered      = False
            trigger_reason = ""

            # --- Priority 1: ball inside bat bounding box ---
            if bat_box is not None and ball_near_bat((x, y), bat_box, margin=35):
                triggered      = True
                trigger_reason = f"BatBox ball=({x},{y})"

            # --- Priority 2: trajectory direction change ---
            elif len(trajectory_history) >= 12:
                curvature   = detect_trajectory_curvature(
                    list(trajectory_history), window_size=5
                )
                accel_spike = detect_acceleration_spike(
                    list(speed_history), window=3
                )

                if curvature > 10 and accel_spike > 1.5 and speed > 5.0:
                    triggered      = True
                    trigger_reason = (
                        f"Traj curv={curvature:.1f}° "
                        f"accel={accel_spike:.1f} spd={speed:.1f}"
                    )

            if triggered:
                contact_detected = True
                contact_frame    = frame_idx
                post_contact_points = []
                contact_frames   = list(frame_buffer)[-30:]
                print(f"🏏 CONTACT [F{frame_idx}] {trigger_reason}")

        # Debug: every 20 frames
        if frame_idx % 20 == 0:
            bat_status = f"bat=✅" if bat_box else "bat=❌"
            ball_status = f"ball=✅({len(detections)})" if detections else "ball=❌"
            print(
                f"[F{frame_idx:4d}] Spd={speed:5.1f} | "
                f"{ball_status} | {bat_status} | "
                f"tracker={'on' if tracker.initialized else 'off'} | "
                f"Events={len(events)}"
            )

        # =================================================
        # STORE POST CONTACT TRAJECTORY
        # =================================================

        if contact_detected:

            frames_since_contact = frame_idx - contact_frame

            if frames_since_contact < 45:

                # Collect post-contact raw frames for shot classifier
                contact_frames.append(frame.copy())

                # Only append if tracker was actually updated this frame
                if tracker_updated:
                    post_contact_points.append((x, y))

            else:
                # Window just closed — run analysis now with all collected points
                contact_detected = False

                # =================================================
                # ANALYZE SHOT (runs once, after full 45-frame window)
                # =================================================

                # Filter out duplicate/stale positions
                unique_points = []
                for pt in post_contact_points:
                    if not unique_points or (
                        abs(pt[0] - unique_points[-1][0]) > 2 or
                        abs(pt[1] - unique_points[-1][1]) > 2
                    ):
                        unique_points.append(pt)

                if len(unique_points) > 4:

                    smooth_points = smooth_trajectory(unique_points)

                    if len(smooth_points) > 5:

                        p1 = smooth_points[0]
                        p2 = smooth_points[-1]

                        dx = p2[0] - p1[0]
                        dy = p1[1] - p2[1]

                        if abs(dx) < 3 and abs(dy) < 3:
                            print(
                                f"[F{frame_idx}] Zero trajectory after window "
                                f"(dx={dx}, dy={dy}, pts={len(unique_points)}) - skip"
                            )
                        else:
                            shot_angle = math.degrees(math.atan2(dy, dx))
                            if shot_angle < 0:
                                shot_angle += 360

                            if shot_classifier is not None and len(contact_frames) >= 10:
                                shot_name, clf_conf = shot_classifier.classify(contact_frames)
                                print(
                                    f"[F{frame_idx}] EfficientNet -> {shot_name} "
                                    f"({clf_conf:.1f}%)  frames={len(contact_frames)}"
                                )
                            else:
                                shot_name = classify_shot(shot_angle)
                                clf_conf  = 0.0
                                print(
                                    f"[F{frame_idx}] geometry -> {shot_name} "
                                    f"angle={shot_angle:.1f} deg"
                                )

                            predicted_distance = 0
                            for i in range(1, len(smooth_points)):
                                px1, py1 = smooth_points[i - 1]
                                px2, py2 = smooth_points[i]
                                predicted_distance += math.sqrt(
                                    (px2 - px1) ** 2 + (py2 - py1) ** 2
                                ) * METERS_PER_PIXEL

                            pts = np.array(smooth_points)
                            try:
                                x_vals = pts[:, 0]
                                y_vals = pts[:, 1]
                                if np.max(x_vals) - np.min(x_vals) > 20:
                                    coeffs = np.polyfit(x_vals, y_vals, 2)
                                    poly = np.poly1d(coeffs)
                                    xs = np.linspace(x_vals[-1], x_vals[-1] + 250, 40)
                                    ys = poly(xs)
                                    future_trajectory = [
                                        (int(fpx), int(fpy))
                                        for fpx, fpy in zip(xs, ys)
                                    ]
                            except Exception:
                                pass

                            already_saved = any(
                                evt.get("frame") == contact_frame and
                                evt.get("ball_id") == ball_id
                                for evt in events
                            )

                            if not already_saved:
                                event = {
                                    "ball_id": ball_id,
                                    "event": "ball_bat_contact",
                                    "frame": contact_frame,
                                    "timestamp_sec": round(contact_frame / fps, 2),
                                    "wagon_wheel_angle": round(shot_angle, 2),
                                    "shot_name": shot_name,
                                    "classifier_confidence": round(clf_conf, 2),
                                    "predicted_distance_m": round(predicted_distance, 2),
                                    "contact_point": {
                                        "x": int(p1[0]),
                                        "y": int(p1[1])
                                    },
                                    "future_trajectory": [
                                        [fpx, fpy] for fpx, fpy in future_trajectory
                                    ]
                                }
                                events.append(event)
                                print("SHOT EVENT:", event)

                else:
                    print(
                        f"[F{frame_idx}] Not enough unique post-contact points "
                        f"({len(unique_points)}) - skip"
                    )

                post_contact_points = []
                contact_frames = []

        # =================================================
        # DRAW TRAJECTORY HISTORY
        # =================================================

        for i in range(
            1,
            len(trajectory_history)
        ):

            cv2.line(
                frame,
                trajectory_history[i - 1],
                trajectory_history[i],
                (255, 255, 0),
                2
            )

        # =================================================
        # DRAW FUTURE TRAJECTORY
        # =================================================

        for i in range(
            1,
            len(future_trajectory)
        ):

            cv2.line(
                frame,
                future_trajectory[i - 1],
                future_trajectory[i],
                (255, 0, 255),
                2
            )

        # =================================================
        # DRAW BALL
        # =================================================

        cv2.circle(
            frame,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        # =================================================
        # TEXT
        # =================================================

        cv2.putText(
            frame,
            f"Speed: {speed:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if shot_name:

            cv2.putText(
                frame,
                f"{shot_name}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Angle: {shot_angle:.1f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Distance: "
                f"{predicted_distance:.1f}m",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2
            )

    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {display_fps}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Cricket Shot Intelligence",
        frame
    )

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================================================
# SAVE
# =========================================================

cap.release()

out.release()

cv2.destroyAllWindows()

with open(EVENTS_JSON, "w") as f:

    json.dump(
        events,
        f,
        indent=4
    )

print(
    f"✅ Saved "
    f"{len(events)} events → "
    f"{EVENTS_JSON}"
)

print(
    f"✅ Saved video → "
    f"{OUTPUT_VIDEO}"
)