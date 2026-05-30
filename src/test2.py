import cv2
import time
import json
import math
import numpy as np
import warnings

from src.detection.yolo_detector import YoloBallDetector
from src.detection.bat_detector import BatDetector

from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball


# =========================================================
# CONFIG
# =========================================================

BALL_MODEL_PATH = (
    r"C:\Cricket-Angle\ball_test\weights\best.pt"
)

BAT_MODEL_PATH = (
    r"C:\Cricket-Train\Cricket Bat detection"
    r"\runs\detect\runs\detect"
    r"\bat_detector_v8n\weights\best.pt"
)

VIDEO_PATH = r"C:\Cricket-Angle\videoplayback.mp4"

EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520


# =========================================================
# HELPERS
# =========================================================

def perspective_scale(y, h):
    return 1.0 + 0.6 * (y / h)


def calculate_angle(p1, p2):

    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    if angle < 0:
        angle += 360

    return angle


def classify_shot(angle):

    if 60 <= angle < 120:
        return "Straight Drive"

    elif 120 <= angle < 160:
        return "Cover Drive"

    elif 160 <= angle < 210:
        return "Pull Shot"

    elif 210 <= angle < 260:
        return "Sweep"

    elif 260 <= angle < 320:
        return "Leg Glance"

    else:
        return "Cut Shot"


# =========================================================
# INIT MODELS
# =========================================================

ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.25,
    ball_class_id=0
)

bat_detector = BatDetector(
    model_path=BAT_MODEL_PATH,
    conf=0.25,
    bat_class_id=1
)

tracker = BallTracker()

cap = cv2.VideoCapture(VIDEO_PATH)

assert cap.isOpened(), "❌ Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS) or 30


# =========================================================
# GLOBAL STATE
# =========================================================

ball_id = 0

events = []

future_trajectory = []

shot_angle = None
shot_name = None
predicted_distance = 0

fps_frames = 0
fps_time = time.time()
display_fps = 0

frame_idx = 0

# ---------------------------------------------------------
# BAT MEMORY
# ---------------------------------------------------------

last_bat_box = None
last_bat_center = None
missing_bat_frames = 0

# ---------------------------------------------------------
# CONTACT STATE
# ---------------------------------------------------------

last_contact_frame = -999

shot_active = False
contact_frame = -999

# ---------------------------------------------------------
# TRAJECTORY STORAGE
# ---------------------------------------------------------

post_contact_points = []


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    # =====================================================
    # FPS
    # =====================================================

    fps_frames += 1

    if time.time() - fps_time >= 1.0:

        display_fps = fps_frames

        fps_frames = 0

        fps_time = time.time()

    # =====================================================
    # BALL DETECTION
    # =====================================================

    predicted = tracker.predict() if tracker.initialized else None

    ball_detections = ball_detector.detect(
        frame,
        predicted
    )

    # -----------------------------------------------------
    # DRAW BALL
    # -----------------------------------------------------

    for cx, cy, x1, y1, x2, y2, conf in ball_detections:

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
    # BAT DETECTION
    # =====================================================

    bat_detections = bat_detector.detect(frame)

    # -----------------------------------------------------
    # FIND BAT CLOSEST TO BALL
    # -----------------------------------------------------

    best_bat = None
    best_distance = 999999

    if (
        tracker.initialized
        and len(bat_detections) > 0
    ):

        ball_x, ball_y = tracker.get_position()

        for det in bat_detections:

            cx, cy, x1, y1, x2, y2, conf = det

            dist = math.sqrt(
                (cx - ball_x) ** 2 +
                (cy - ball_y) ** 2
            )

            # ignore detections far away
            if dist > 350:
                continue

            if dist < best_distance:

                best_distance = dist
                best_bat = det

    # -----------------------------------------------------
    # STORE BAT
    # -----------------------------------------------------

    if best_bat is not None:

        cx, cy, x1, y1, x2, y2, conf = best_bat

        last_bat_box = (x1, y1, x2, y2)
        last_bat_center = (cx, cy)

        missing_bat_frames = 0

    else:

        missing_bat_frames += 1

    # -----------------------------------------------------
    # KEEP BAT FOR FEW FRAMES
    # -----------------------------------------------------

    if (
        last_bat_box is not None
        and missing_bat_frames < 8
    ):

        x1, y1, x2, y2 = last_bat_box

        bat_center = last_bat_center

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.circle(
            frame,
            bat_center,
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            "BAT",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    else:

        bat_center = None

    # =====================================================
    # TRACKER UPDATE
    # =====================================================

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

    # =====================================================
    # CONTACT DETECTION
    # =====================================================

    if (
        tracker.initialized
        and bat_center is not None
        and last_bat_box is not None
    ):

        ball_x, ball_y = tracker.get_position()

        x1, y1, x2, y2 = last_bat_box

        # -------------------------------------------------
        # BALL BOX
        # -------------------------------------------------

        ball_box = (
            ball_x - 8,
            ball_y - 8,
            ball_x + 8,
            ball_y + 8
        )

        # -------------------------------------------------
        # INTERSECTION TEST
        # -------------------------------------------------

        intersects = not (

            ball_box[2] < x1 or
            ball_box[0] > x2 or
            ball_box[3] < y1 or
            ball_box[1] > y2
        )

        # -------------------------------------------------
        # START SHOT ONLY ONCE
        # -------------------------------------------------

        if (
            intersects and
            not shot_active and
            (frame_idx - last_contact_frame) > 30
        ):

            print("🏏 CONTACT DETECTED")

            shot_active = True

            last_contact_frame = frame_idx

            contact_frame = frame_idx

            post_contact_points = []

    # =====================================================
    # STORE POST CONTACT TRAJECTORY
    # =====================================================

    if (
        shot_active and
        tracker.initialized
    ):

        if (
            frame_idx - contact_frame
        ) < 25:

            bx, by = tracker.get_position()

            post_contact_points.append((bx, by))

        else:

            shot_active = False

    # =====================================================
    # ANALYZE SHOT
    # =====================================================

    if len(post_contact_points) > 10:

        p1 = post_contact_points[0]
        p2 = post_contact_points[-1]

        # -------------------------------------------------
        # CHECK MOVEMENT
        # -------------------------------------------------

        motion = math.sqrt(
            (p2[0] - p1[0]) ** 2 +
            (p2[1] - p1[1]) ** 2
        )

        # ignore static trajectories
        if motion > 25:

            # -------------------------------------------------
            # ANGLE
            # -------------------------------------------------

            shot_angle = calculate_angle(p1, p2)

            shot_name = classify_shot(
                shot_angle
            )

            # -------------------------------------------------
            # DISTANCE
            # -------------------------------------------------

            predicted_distance = 0

            for i in range(
                1,
                len(post_contact_points)
            ):

                x1, y1 = post_contact_points[i - 1]
                x2, y2 = post_contact_points[i]

                dist = math.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

                predicted_distance += (
                    dist * METERS_PER_PIXEL
                )

            # -------------------------------------------------
            # POLYFIT TRAJECTORY
            # -------------------------------------------------

            pts = np.array(post_contact_points)

            if len(pts) > 8:

                try:

                    x = pts[:, 0]
                    y = pts[:, 1]

                    x_range = np.max(x) - np.min(x)
                    y_range = np.max(y) - np.min(y)

                    # avoid bad polyfit
                    if x_range > 15 or y_range > 15:

                        with warnings.catch_warnings():

                            warnings.simplefilter(
                                "ignore",
                                np.RankWarning
                            )

                            coeffs = np.polyfit(
                                x,
                                y,
                                2
                            )

                        poly = np.poly1d(coeffs)

                        xs = np.linspace(
                            x.min(),
                            x.max(),
                            50
                        )

                        ys = poly(xs)

                        future_trajectory = []

                        for i in range(len(xs)):

                            px = int(xs[i])
                            py = int(ys[i])

                            future_trajectory.append(
                                (px, py)
                            )

                except:
                    pass

            # -------------------------------------------------
            # SAVE EVENT
            # -------------------------------------------------

            if (
                len(events) == 0 or
                events[-1]["frame"] != frame_idx
            ):

                event = {

                    "ball_id": ball_id,

                    "event": "ball_bat_contact",

                    "frame": frame_idx,

                    "timestamp_sec": round(
                        frame_idx / fps,
                        2
                    ),

                    "wagon_wheel_angle": round(
                        shot_angle,
                        2
                    ),

                    "shot_name": shot_name,

                    "predicted_distance_m": round(
                        predicted_distance,
                        2
                    ),

                    "trajectory_points": [
                        [int(px), int(py)]
                        for px, py in post_contact_points
                    ]
                }

                events.append(event)

                print(
                    "🏏 SHOT:",
                    shot_name,
                    "| Angle:",
                    shot_angle
                )

            # prevent repeated event spam
            shot_active = False

    # =====================================================
    # DRAW TRAJECTORY
    # =====================================================

    if len(future_trajectory) > 1:

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

    # =====================================================
    # TRACKER VISUALS
    # =====================================================

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

        if shot_name:

            cv2.putText(
                frame,
                f"{shot_name} | {shot_angle:.1f} deg",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {display_fps}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "CricketSense | Shot Intelligence",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================================================
# SAVE JSON
# =========================================================

cap.release()
cv2.destroyAllWindows()

with open(EVENTS_JSON, "w") as f:
    json.dump(events, f, indent=4)

print(
    f"✅ Saved {len(events)} events → "
    f"{EVENTS_JSON}"
)