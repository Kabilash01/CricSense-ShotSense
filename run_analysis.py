"""
run_analysis.py
===============
Live shot analysis on a cricket video using OpenCV display.

Two separate YOLO models
------------------------
  Ball detector : YOLOv8  (ball_test/weights/best.pt)   class 0 = ball
  Bat  detector : YOLOv5  (models/weights/best.pt)       class 1 = bat

Overlays per frame
------------------
  - Ball detection bbox (green) + tracked position (red dot)
  - Ball trajectory trail (orange)
  - Bat detection bbox (yellow)
  - Release point marker (orange circle)
  - Speed / Release speed / Max speed
  - Pitch zone colour band (transparent)
  - CONTACT flash + vertical line (cyan)
  - Predicted 2-second ground trajectory (green dashes)
  - Shot angle (deg) and shot name (bottom-left, persists after contact)

Controls
--------
  q  quit
  r  force end-of-delivery analysis and reset
  s  save current frame as PNG

Usage
-----
  python run_analysis.py
  python run_analysis.py --video path/to/clip.mp4
  python run_analysis.py --no-display   # headless, writes output video only
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config — edit or override via CLI
# ---------------------------------------------------------------------------
BALL_MODEL_PATH = r"ball_test/weights/best.pt"   # YOLOv8: class 0=ball
BAT_MODEL_PATH  = r"models/weights/best.pt"      # YOLOv5: class 0=ball, 1=bat
VIDEO_PATH      = r"C:/Cricket-Angle/videoplayback.mp4"
OUTPUT_VIDEO    = r"outputs/videos/shot_analysis_output.mp4"
OUTPUT_JSON     = r"outputs/json/shot_analysis.json"
HANDEDNESS      = "right"

BALL_CLASS_ID    = 0
BAT_CLASS_ID     = 1
BALL_CONF        = 0.20
BAT_CONF         = 0.25
METERS_PER_PIXEL = 18.5 / 520

PROXIMITY_PX        = 30
ANGLE_CHANGE_DEG    = 25
CONTACT_WINDOW      = 3
INTERP_MAX_GAP      = 3
POST_CONTACT_FRAMES = 60
DIRECTION_LOOKAHEAD = 10

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video",      default=VIDEO_PATH)
parser.add_argument("--ball-model", default=BALL_MODEL_PATH)
parser.add_argument("--bat-model",  default=BAT_MODEL_PATH)
parser.add_argument("--output",     default=OUTPUT_VIDEO)
parser.add_argument("--handedness", default=HANDEDNESS)
parser.add_argument("--no-display", action="store_true")
args = parser.parse_args()

VIDEO_PATH      = args.video
BALL_MODEL_PATH = args.ball_model
BAT_MODEL_PATH  = args.bat_model
OUTPUT_VIDEO    = args.output
HANDEDNESS      = args.handedness
HEADLESS        = args.no_display

# ---------------------------------------------------------------------------
# src/ imports (tracker, association)
# ---------------------------------------------------------------------------
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------------------------------------------------------------------
# shot_analyzer imports
# ---------------------------------------------------------------------------
from shot_analyzer import (
    _safe_yolov8_load,
    _safe_yolo_load,
    detect_contact,
    compute_shot_angle,
    classify_shot,
    ground_to_image,
    fit_and_predict_ground,
    compute_homography,
)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
def _model_is_valid(path):
    return os.path.isfile(path) and os.path.getsize(path) > 1024

print("[init] loading bat model (YOLOv5) …")
_bat_hub = _safe_yolo_load(BAT_MODEL_PATH)
print("[init] bat model classes:", _bat_hub.names)

# Ball model: prefer a dedicated YOLOv8 checkpoint; fall back to YOLOv5 best.pt
_ball_is_yolov8 = _model_is_valid(BALL_MODEL_PATH)
if _ball_is_yolov8:
    print("[init] loading ball model (YOLOv8) …")
    _ball_model = _safe_yolov8_load(BALL_MODEL_PATH)
    print("[init] ball model classes:", _ball_model.names)
else:
    print(f"[init] {BALL_MODEL_PATH} missing/empty — using YOLOv5 bat model for ball (class 0)")
    _ball_model = _bat_hub   # same model, filter by class 0


# ---------------------------------------------------------------------------
# Ball detector
# ---------------------------------------------------------------------------
class _BallDetector:
    """
    Detects ball in a frame.
    Supports both YOLOv8 (ultralytics YOLO) and YOLOv5 (_YOLOv5Wrapper).
    Applies area + aspect-ratio filters to reduce false positives.
    """
    def __init__(self, model, conf, class_id, is_yolov8: bool):
        self._m        = model
        self.conf      = conf
        self.class_id  = class_id
        self._is_v8    = is_yolov8

    def detect(self, frame, _=None):
        if self._is_v8:
            results = self._m(frame, conf=self.conf, verbose=False)
            if not results or results[0].boxes is None:
                return []
            boxes_obj = results[0].boxes
            raw = zip(boxes_obj.xyxy, boxes_obj.cls, boxes_obj.conf)
        else:
            result = self._m(frame, conf=self.conf)[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
            raw = zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)

        dets = []
        for box, cls, conf in raw:
            if int(cls) != self.class_id:
                continue
            x1, y1, x2, y2 = map(int, box.tolist() if hasattr(box, "tolist") else [float(v) for v in box])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if not (20 <= bw * bh <= 2500):
                continue
            if not (0.6 <= bw / bh <= 1.4):
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dets.append((cx, cy, x1, y1, x2, y2, float(conf)))
        return dets


# ---------------------------------------------------------------------------
# Bat detector — YOLOv5 (via _YOLOv5Wrapper), class 1 = bat
# ---------------------------------------------------------------------------
class _BatDetector:
    """Wraps YOLOv5 hub model; returns the highest-conf bat box per frame."""
    def __init__(self, hub_model, conf, class_id):
        self._m       = hub_model
        self.conf     = conf
        self.class_id = class_id

    def detect(self, frame):
        result = self._m(frame, conf=self.conf)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        best_box, best_c = None, 0.0
        for box, cls, conf in zip(result.boxes.xyxy,
                                   result.boxes.cls,
                                   result.boxes.conf):
            if int(cls) != self.class_id:
                continue
            c = float(conf)
            if c > best_c:
                best_c   = c
                best_box = tuple(int(v) for v in box.tolist())
        return best_box   # (x1, y1, x2, y2) or None


ball_detector = _BallDetector(_ball_model, conf=BALL_CONF, class_id=BALL_CLASS_ID,
                               is_yolov8=_ball_is_yolov8)
bat_detector  = _BatDetector(_bat_hub,     conf=BAT_CONF,  class_id=BAT_CLASS_ID)

# ---------------------------------------------------------------------------
# Open video
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"[error] cannot open video: {VIDEO_PATH}")

FW  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FH  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_VIDEO)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_JSON)),  exist_ok=True)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FW, FH),
)

# ---------------------------------------------------------------------------
# Approximate homography from frame centre strip (no calibration required).
# Angles computed in pixel-space are still correct for shot classification.
# Replace H_mat with calibrate_from_clicks() output for true ground metres.
# ---------------------------------------------------------------------------
_cal_pts = np.array([
    [FW * 0.6, FH * 0.1],
    [FW * 0.4, FH * 0.1],
    [FW * 0.4, FH * 0.9],
    [FW * 0.6, FH * 0.9],
], dtype=np.float64)
H_mat = compute_homography(_cal_pts)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
tracker    = BallTracker(fps=int(FPS))
trail      = deque(maxlen=40)

ball_track: dict = {}
bat_boxes:  dict = {}

contact_event  = None
shot_angle_deg = None
shot_name_str  = None
pred_img_pts   = None
contact_flash  = 0

deliveries  = []
delivery_id = 0
frame_idx   = 0

fps_counter = 0
fps_timer   = time.time()
display_fps = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ZONES = {
    "YORKER": (0.75, 1.00, (0,   0,   255)),
    "FULL":   (0.55, 0.75, (0,   255, 255)),
    "GOOD":   (0.35, 0.55, (0,   255, 0  )),
    "SHORT":  (0.00, 0.35, (255, 0,   0  )),
}

def _draw_zone(frame, ptype):
    if ptype not in _ZONES:
        return
    y1n, y2n, col = _ZONES[ptype]
    ov = frame.copy()
    cv2.rectangle(ov, (0, int(y1n * FH)), (FW, int(y2n * FH)), col, -1)
    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

def _draw_trail(frame, pts):
    pts = list(pts)
    for i in range(1, len(pts)):
        cv2.line(frame,
                 (int(pts[i-1][0]), int(pts[i-1][1])),
                 (int(pts[i][0]),   int(pts[i][1])),
                 (255, 120, 0), 2)

def _txt(frame, text, pos, color=(0, 255, 255), scale=0.75, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

def _reset():
    global ball_track, bat_boxes, contact_event, shot_angle_deg
    global shot_name_str, pred_img_pts, contact_flash, delivery_id
    delivery_id    += 1
    ball_track      = {}
    bat_boxes       = {}
    contact_event   = None
    shot_angle_deg  = None
    shot_name_str   = None
    pred_img_pts    = None
    contact_flash   = 0
    trail.clear()
    tracker.reset()

def _analyse():
    global contact_event, shot_angle_deg, shot_name_str, pred_img_pts, contact_flash

    if len(ball_track) < 2 * CONTACT_WINDOW + 1:
        return

    ce = detect_contact(
        ball_track, bat_boxes,
        proximity_px=PROXIMITY_PX,
        angle_change_deg=ANGLE_CHANGE_DEG,
        window=CONTACT_WINDOW,
        interp_max_gap=INTERP_MAX_GAP,
    )
    if ce is None:
        return

    contact_event = ce
    contact_flash = 45

    observed, predicted = fit_and_predict_ground(
        ball_track, H_mat, ce.frame_idx, FPS,
        post_contact_frames=POST_CONTACT_FRAMES,
    )

    ang = compute_shot_angle(observed, ce.frame_idx,
                             direction_lookahead=DIRECTION_LOOKAHEAD)

    # Pixel-space fallback: if ground projection failed (too few post-contact
    # detections), compute the angle directly from the raw ball_track pixels.
    # Angles are still correct for shot classification — only the ground
    # projection (metres) is inaccurate.
    if ang is None and len(ball_track) >= ce.frame_idx + DIRECTION_LOOKAHEAD:
        sorted_frames = sorted(f for f in ball_track if f >= ce.frame_idx)
        if len(sorted_frames) >= 2:
            f0, f1 = sorted_frames[0], sorted_frames[min(DIRECTION_LOOKAHEAD, len(sorted_frames)-1)]
            dx = ball_track[f1][0] - ball_track[f0][0]
            dy = ball_track[f1][1] - ball_track[f0][1]
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                # pixel Y increases downward; negate dy so +Y means "away from bat"
                ang = float(math.degrees(math.atan2(dx, -dy)) % 360)

    if ang is not None:
        shot_angle_deg = ang
        shot_name_str  = classify_shot(ang, HANDEDNESS)

    if predicted:
        gpts = np.array([(x, y) for (_, x, y) in predicted], dtype=np.float64)
        try:
            ipts = ground_to_image(gpts, H_mat).astype(int)
            mask = ((ipts[:, 0] >= 0) & (ipts[:, 0] < FW) &
                    (ipts[:, 1] >= 0) & (ipts[:, 1] < FH))
            pred_img_pts = ipts[mask]
        except Exception:
            pred_img_pts = None

    deliveries.append({
        "delivery_id":    delivery_id,
        "contact_frame":  int(ce.frame_idx),
        "proximity_px":   round(ce.proximity_px, 1),
        "angle_change":   round(ce.angle_change_deg, 1),
        "shot_angle_deg": round(shot_angle_deg, 1) if shot_angle_deg else None,
        "shot_name":      shot_name_str,
        "handedness":     HANDEDNESS,
    })
    ang_str = f"{shot_angle_deg:.1f}" if shot_angle_deg is not None else "-"
    print(f"[delivery {delivery_id}] contact@{ce.frame_idx}  "
          f"{shot_name_str or 'unknown'}  {ang_str} deg")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print(f"[run] {VIDEO_PATH}  |  q=quit  r=reset  s=save-frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS counter
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        display_fps = fps_counter
        fps_counter = 0
        fps_timer   = time.time()

    # ------------------------------------------------------------------
    # 1. Ball detection (YOLOv8) + Kalman tracking
    # ------------------------------------------------------------------
    pred_pos  = tracker.predict() if tracker.initialized else None
    ball_dets = ball_detector.detect(frame, pred_pos)
    match     = associate_ball(ball_dets, pred_pos)

    if match is not None:
        cx, cy = match[0], match[1]
        tracker.update((cx, cy))
        ball_track[frame_idx] = (float(cx), float(cy))
        trail.append((cx, cy))
        tracker.missed_frames = 0
        x1, y1, x2, y2 = int(match[2]), int(match[3]), int(match[4]), int(match[5])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        tracker.missed_frames += 1

    # ------------------------------------------------------------------
    # 2. Bat detection (YOLOv5)
    # ------------------------------------------------------------------
    bat_box = bat_detector.detect(frame)
    if bat_box is not None:
        bat_boxes[frame_idx] = bat_box
        bx1, by1, bx2, by2 = bat_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
        _txt(frame, "bat", (bx1, max(0, by1 - 6)),
             color=(0, 200, 255), scale=0.5, thick=1)

    # ------------------------------------------------------------------
    # 3. End-of-delivery trigger
    # ------------------------------------------------------------------
    if tracker.missed_frames > 15 and tracker.initialized:
        _analyse()
        _reset()

    # ------------------------------------------------------------------
    # 4. Draw overlays
    # ------------------------------------------------------------------
    _draw_trail(frame, trail)

    if tracker.initialized:
        px, py = tracker.get_position()
        cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 255), -1)

        if tracker.release_point:
            rx, ry = tracker.release_point
            cv2.circle(frame, (rx, ry), 8, (255, 80, 0), -1)
            _txt(frame, "RELEASE", (rx + 8, ry - 8), color=(255, 80, 0), scale=0.55)

        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(FH)
        if tracker.pitch_type:
            _draw_zone(frame, tracker.pitch_type)
            _txt(frame, tracker.pitch_type, (20, 120),
                 color=(0, 255, 0), scale=1.0, thick=3)

        scale = 1.0 + 0.6 * (py / FH)
        speed = tracker.get_speed_kmph(METERS_PER_PIXEL, scale)
        _txt(frame, f"Speed:   {speed:.1f} km/h", (20, 30))
        if tracker.release_speed:
            _txt(frame, f"Release: {tracker.release_speed:.1f} km/h",
                 (20, 58), color=(255, 255, 0))
        _txt(frame, f"Max:     {tracker.max_speed:.1f} km/h",
             (20, 86), color=(0, 80, 255))

    # Predicted trajectory
    if pred_img_pts is not None and len(pred_img_pts) > 1:
        for i in range(1, len(pred_img_pts)):
            cv2.line(frame, tuple(pred_img_pts[i-1]),
                     tuple(pred_img_pts[i]), (0, 255, 100), 2)

    # Contact flash
    if contact_flash > 0:
        cv2.line(frame, (FW // 2, 0), (FW // 2, FH), (0, 255, 255), 1)
        _txt(frame, "CONTACT!", (FW // 2 - 90, 55),
             color=(0, 255, 255), scale=1.3, thick=3)
        contact_flash -= 1

    # Shot label
    if shot_name_str:
        _txt(frame, f"Angle: {shot_angle_deg:.1f} deg",
             (20, FH - 60), scale=0.8)
        _txt(frame, f"Shot:  {shot_name_str}",
             (20, FH - 30), scale=0.9)

    # FPS
    _txt(frame, f"FPS {display_fps}", (FW - 110, 28),
         color=(220, 220, 220), scale=0.7, thick=1)

    # ------------------------------------------------------------------
    # 5. Write + display
    # ------------------------------------------------------------------
    writer.write(frame)

    if not HEADLESS:
        cv2.imshow("CricSense — Shot Analysis  (q=quit  r=reset  s=save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            _analyse()
            _reset()
        elif key == ord("s"):
            fname = f"frame_{frame_idx:05d}.png"
            cv2.imwrite(fname, frame)
            print(f"[saved] {fname}")

    frame_idx += 1

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
_analyse()   # catch any in-progress delivery at end-of-file

cap.release()
writer.release()
if not HEADLESS:
    cv2.destroyAllWindows()

with open(OUTPUT_JSON, "w") as f:
    json.dump(deliveries, f, indent=2)

print(f"\n[done] {frame_idx} frames | {len(deliveries)} deliveries")
print(f"       video -> {OUTPUT_VIDEO}")
print(f"       JSON  -> {OUTPUT_JSON}")
