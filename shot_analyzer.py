"""
shot_analyzer.py
================

Shot analysis pipeline for CricSense / Cricket-Angle.

Adds the following on top of the existing YOLO ball+bat detector and tracker:

  1. Bat-ball contact detection (proximity + velocity-angle fusion, with
     short-gap interpolation of the ball track to survive motion blur).
  2. Ground-plane homography (4-click calibration helper) and projection.
  3. Post-contact ground trajectory fit + 1-2 second prediction.
  4. Shot angle (wagon-wheel angle, clockwise from +Y / "toward bowler").
  5. Rule-based shot-name classification, with handedness handling and a
     clean hook for a learned LSTM head later.
  6. End-to-end wrapper `analyse_delivery(...)`.
  7. Video overlay rendering.

Design notes
------------
* Pure NumPy / OpenCV. No new heavy dependencies.
* Detection + tracking are NOT re-implemented; you pass in the ball track and
  bat boxes you already obtain from `src/detection/yolo_detector.py` and
  `src/tracking/tracker.py`.
* Notebook-friendly: import this single file and call the functions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cricket pitch dimensions (ICC):
#   22 yards long  = 20.12 m
#   10 ft  wide    =  3.05 m
PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05

# Ground reference frame:
#   origin   = striker's wicket centre
#   +Y axis  = toward bowler (down the pitch)
#   +X axis  = off side for a right-handed batsman
#
# Click order during calibration:
#   1. striker-end off  corner   -> (+X/2, 0)
#   2. striker-end leg  corner   -> (-X/2, 0)
#   3. bowler-end  leg  corner   -> (-X/2, +Y)
#   4. bowler-end  off  corner   -> (+X/2, +Y)
GROUND_REF_POINTS = np.array(
    [
        [+PITCH_WIDTH_M / 2.0, 0.0],
        [-PITCH_WIDTH_M / 2.0, 0.0],
        [-PITCH_WIDTH_M / 2.0, PITCH_LENGTH_M],
        [+PITCH_WIDTH_M / 2.0, PITCH_LENGTH_M],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContactEvent:
    """Single bat-ball contact event."""
    frame_idx: int
    ball_xy: Tuple[float, float]
    bat_box: Optional[Tuple[int, int, int, int]]
    angle_change_deg: float
    proximity_px: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "ball_xy": [float(self.ball_xy[0]), float(self.ball_xy[1])],
            "bat_box": list(self.bat_box) if self.bat_box is not None else None,
            "angle_change_deg": float(self.angle_change_deg),
            "proximity_px": float(self.proximity_px),
            "confidence": float(self.confidence),
        }


@dataclass
class ShotResult:
    """Output of `analyse_delivery`."""
    contact: Optional[ContactEvent]
    observed_ground: List[Tuple[int, float, float]] = field(default_factory=list)
    predicted_ground: List[Tuple[float, float, float]] = field(default_factory=list)
    angle_deg: Optional[float] = None
    shot_name: Optional[str] = None
    handedness: str = "right"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "contact": self.contact.to_dict() if self.contact else None,
            "observed_ground": [
                {"frame": int(f), "x_m": float(x), "y_m": float(y)}
                for (f, x, y) in self.observed_ground
            ],
            "predicted_ground": [
                {"t_s": float(t), "x_m": float(x), "y_m": float(y)}
                for (t, x, y) in self.predicted_ground
            ],
            "angle_deg": None if self.angle_deg is None else float(self.angle_deg),
            "shot_name": self.shot_name,
            "handedness": self.handedness,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 0. Track interpolation (handles motion-blur drops at contact)
# ---------------------------------------------------------------------------

def interpolate_track(
    track: Dict[int, Tuple[float, float]],
    max_gap: int = 3,
) -> Dict[int, Tuple[float, float]]:
    """
    Fill missed frames in a ball track by linear interpolation when the gap is
    at most `max_gap` frames. Gaps larger than `max_gap` are left as holes.

    Parameters
    ----------
    track : dict[int, (x, y)]
        Sparse mapping frame_idx -> ball pixel position.
    max_gap : int
        Maximum allowed missed-frame gap to interpolate over.

    Returns
    -------
    dict[int, (x, y)]
        Dense (within tolerated gaps) mapping.
    """
    if not track:
        return {}

    frames = sorted(track.keys())
    out: Dict[int, Tuple[float, float]] = {f: track[f] for f in frames}

    for a, b in zip(frames[:-1], frames[1:]):
        gap = b - a
        if gap <= 1 or gap - 1 > max_gap:
            continue

        (xa, ya) = track[a]
        (xb, yb) = track[b]
        for i in range(1, gap):
            t = i / gap
            out[a + i] = (xa + t * (xb - xa), ya + t * (yb - ya))

    return out


# ---------------------------------------------------------------------------
# 1. Bat-ball contact detection
# ---------------------------------------------------------------------------

def _bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_to_bbox_distance(
    pt: Tuple[float, float], box: Sequence[float]
) -> float:
    """Shortest distance from a point to a (possibly distant) bbox."""
    x, y = pt
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _velocity(
    track: Dict[int, Tuple[float, float]], f0: int, f1: int
) -> Optional[Tuple[float, float]]:
    """Average velocity (per-frame, in px) between two frames if both exist."""
    if f0 not in track or f1 not in track or f0 == f1:
        return None
    x0, y0 = track[f0]
    x1, y1 = track[f1]
    return ((x1 - x0) / (f1 - f0), (y1 - y0) / (f1 - f0))


def _angle_between(
    v1: Tuple[float, float], v2: Tuple[float, float]
) -> float:
    """Unsigned angle between two 2-D vectors in degrees."""
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def detect_contact(
    ball_track: Dict[int, Tuple[float, float]],
    bat_boxes: Dict[int, Tuple[int, int, int, int]],
    *,
    proximity_px: float = 30.0,
    angle_change_deg: float = 25.0,
    window: int = 3,
    interp_max_gap: int = 3,
) -> Optional[ContactEvent]:
    """
    Detect the first frame where bat-ball contact occurs.

    A frame `f` is a contact iff BOTH cues fire:
      (a) Ball pixel position is within `proximity_px` of the bat bbox at `f`
          (using point-to-bbox shortest distance).
      (b) The angle between the average ball velocity over [f-window, f] and
          [f, f+window] exceeds `angle_change_deg`.

    Short gaps in the ball track (<= `interp_max_gap` frames) are linearly
    interpolated first, because motion blur at contact frequently kills YOLO
    detection at the exact frame that matters.

    Returns
    -------
    ContactEvent | None
    """
    if not ball_track or not bat_boxes:
        return None

    track = interpolate_track(ball_track, max_gap=interp_max_gap)
    frames = sorted(track.keys())
    if len(frames) < 2 * window + 1:
        return None

    # Pre-sort bat frames so we can pick the nearest available bat box.
    bat_frames = sorted(bat_boxes.keys())

    def _nearest_bat(f: int) -> Optional[Tuple[int, int, int, int]]:
        """Bat box at frame f, or the closest one within +/- window frames."""
        if f in bat_boxes:
            return bat_boxes[f]
        best = None
        best_d = window + 1
        for bf in bat_frames:
            d = abs(bf - f)
            if d <= window and d < best_d:
                best = bat_boxes[bf]
                best_d = d
        return best

    for f in frames:
        if (f - window) not in track or (f + window) not in track:
            continue

        v_pre = _velocity(track, f - window, f)
        v_post = _velocity(track, f, f + window)
        if v_pre is None or v_post is None:
            continue

        bat = _nearest_bat(f)
        if bat is None:
            continue

        prox = _point_to_bbox_distance(track[f], bat)
        if prox > proximity_px:
            continue

        ang = _angle_between(v_pre, v_post)
        if ang < angle_change_deg:
            continue

        # Confidence: combine how decisively the two thresholds were exceeded.
        prox_score = max(0.0, 1.0 - prox / max(proximity_px, 1e-6))
        ang_score = min(1.0, ang / 90.0)
        confidence = 0.5 * prox_score + 0.5 * ang_score

        return ContactEvent(
            frame_idx=int(f),
            ball_xy=(float(track[f][0]), float(track[f][1])),
            bat_box=tuple(int(v) for v in bat),
            angle_change_deg=float(ang),
            proximity_px=float(prox),
            confidence=float(confidence),
        )

    return None


# ---------------------------------------------------------------------------
# 2. Ground-plane homography
# ---------------------------------------------------------------------------

def _grab_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    return frame


def calibrate_from_clicks(
    video_path: str,
    save_path: Optional[str] = None,
    window_name: str = "Calibrate pitch corners",
) -> np.ndarray:
    """
    Interactive 4-click calibration on the first frame of `video_path`.

    Click order:
      1) striker-end OFF corner
      2) striker-end LEG corner
      3) bowler-end  LEG corner
      4) bowler-end  OFF corner

    Press `r` to reset, `q` to abort.

    Returns
    -------
    H : (3,3) np.ndarray
        Homography mapping image pixels -> ground metres.
    """
    frame = _grab_first_frame(video_path)
    clone = frame.copy()
    clicks: List[Tuple[int, int]] = []
    labels = [
        "1) striker OFF corner",
        "2) striker LEG corner",
        "3) bowler  LEG corner",
        "4) bowler  OFF corner",
    ]

    def _on_mouse(event, x, y, flags, _param):
        nonlocal clone
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                clone, str(len(clicks)), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        display = clone.copy()
        instruction = (
            labels[len(clicks)] if len(clicks) < 4
            else "All 4 clicked. Press any key to confirm."
        )
        cv2.putText(
            display, instruction, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration aborted by user")
        if key == ord("r"):
            clicks.clear()
            clone = frame.copy()
            continue
        if len(clicks) == 4 and key != 255:
            break

    cv2.destroyWindow(window_name)

    H = compute_homography(np.array(clicks, dtype=np.float64))
    if save_path:
        save_homography(H, save_path)
    return H


def compute_homography(image_points: np.ndarray) -> np.ndarray:
    """
    Compute pixel-to-ground homography from 4 image points clicked in the
    canonical order (see `calibrate_from_clicks`).
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(4, 2)
    H, status = cv2.findHomography(image_points, GROUND_REF_POINTS, 0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed to find a homography")
    return H


def save_homography(H: np.ndarray, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"H": np.asarray(H).tolist()}, f, indent=2)


def load_homography(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float64)


def image_to_ground(
    uv: Sequence[Tuple[float, float]] | np.ndarray, H: np.ndarray
) -> np.ndarray:
    """
    Project image pixel points to ground metres using homography H.

    Parameters
    ----------
    uv : sequence of (x, y) pixel coords, or (N,2) array
    H  : (3,3) homography (pixels -> metres)

    Returns
    -------
    (N, 2) array of ground (X_m, Y_m) coordinates.
    """
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def ground_to_image(
    xy: Sequence[Tuple[float, float]] | np.ndarray, H: np.ndarray
) -> np.ndarray:
    """Project ground metres back to image pixels via H^-1."""
    Hinv = np.linalg.inv(H)
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, Hinv)
    return out.reshape(-1, 2)


# ---------------------------------------------------------------------------
# 3. Post-contact ground trajectory + prediction
# ---------------------------------------------------------------------------

def _collect_post_contact(
    track: Dict[int, Tuple[float, float]],
    contact_frame: int,
    post_contact_frames: int,
) -> List[Tuple[int, Tuple[float, float]]]:
    out = []
    for f in range(contact_frame, contact_frame + post_contact_frames + 1):
        if f in track:
            out.append((f, track[f]))
    return out


def fit_and_predict_ground(
    ball_track: Dict[int, Tuple[float, float]],
    H: np.ndarray,
    contact_frame: int,
    fps: float = 30.0,
    *,
    post_contact_frames: int = 60,
    prediction_horizon_s: float = 2.0,
    prediction_step_s: float = 0.05,
    interp_max_gap: int = 3,
) -> Tuple[List[Tuple[int, float, float]], List[Tuple[float, float, float]]]:
    """
    Project post-contact ball pixel positions to ground (m), fit a degree-2
    polynomial to X(t) and Y(t) separately, then predict for t in
    [0, prediction_horizon_s].

    Returns
    -------
    observed_ground : list[(frame_idx, X_m, Y_m)]
    predicted_ground : list[(t_s, X_m, Y_m)]
    """
    track = interpolate_track(ball_track, max_gap=interp_max_gap)
    samples = _collect_post_contact(track, contact_frame, post_contact_frames)
    if len(samples) < 5:
        return [], []

    frames = np.array([s[0] for s in samples], dtype=np.float64)
    pix = np.array([s[1] for s in samples], dtype=np.float64)
    ground = image_to_ground(pix, H)

    observed = [
        (int(f), float(g[0]), float(g[1]))
        for f, g in zip(frames, ground)
    ]

    t = (frames - contact_frame) / float(fps)

    # Degree-2 polynomial: linear motion + deceleration / lateral curve.
    px_coeffs = np.polyfit(t, ground[:, 0], deg=2)
    py_coeffs = np.polyfit(t, ground[:, 1], deg=2)
    px = np.poly1d(px_coeffs)
    py = np.poly1d(py_coeffs)

    t_pred = np.arange(0.0, prediction_horizon_s + 1e-9, prediction_step_s)
    predicted = [
        (float(ti), float(px(ti)), float(py(ti)))
        for ti in t_pred
    ]
    return observed, predicted


# ---------------------------------------------------------------------------
# 4. Shot angle
# ---------------------------------------------------------------------------

def compute_shot_angle(
    observed_ground: List[Tuple[int, float, float]],
    contact_frame: int,
    *,
    direction_lookahead: int = 10,
    batsman_xy: Tuple[float, float] = (0.0, 0.0),
) -> Optional[float]:
    """
    Wagon-wheel angle of the shot, measured CLOCKWISE from +Y axis
    (i.e. from "straight toward the bowler") in degrees [0, 360).

    Uses ground coordinates. The direction vector goes from `batsman_xy`
    (default = origin = striker's wicket centre) to the ball's ground
    position roughly `direction_lookahead` frames after contact.

    Returns None if there aren't enough post-contact samples.
    """
    if not observed_ground:
        return None

    # Pick the sample closest to (contact_frame + direction_lookahead).
    target = contact_frame + direction_lookahead
    obs_sorted = sorted(observed_ground, key=lambda r: abs(r[0] - target))
    _, x, y = obs_sorted[0]

    dx = x - batsman_xy[0]
    dy = y - batsman_xy[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    # atan2(dx, dy) gives angle measured CW from +Y, in radians.
    angle = math.degrees(math.atan2(dx, dy))
    return float(angle % 360.0)


# ---------------------------------------------------------------------------
# 5. Shot classification
# ---------------------------------------------------------------------------

# (low_deg, high_deg, name). Wrap-around bin handled separately.
_SHOT_TABLE_RIGHT: List[Tuple[float, float, str]] = [
    (15.0, 40.0, "Off drive"),
    (40.0, 75.0, "Cover drive"),
    (75.0, 105.0, "Square cut"),
    (105.0, 150.0, "Late cut / Upper cut"),
    (150.0, 200.0, "Edge (slip / keeper area)"),
    (200.0, 250.0, "Leg glance / Sweep"),
    (250.0, 285.0, "Pull"),
    (285.0, 320.0, "Mid-wicket pull / Flick"),
    (320.0, 345.0, "On drive"),
]


def classify_shot(
    angle_deg: float,
    handedness: str = "right",
) -> str:
    """
    Rule-based shot-name lookup from wagon-wheel angle.

    For left-handed batters, mirror around the vertical axis with
    (360 - angle) % 360 before lookup.
    """
    if angle_deg is None:
        return "Unknown"

    a = float(angle_deg) % 360.0
    if handedness.lower().startswith("l"):
        a = (360.0 - a) % 360.0

    # Wrap-around bin for the "straight drive" cone 345..360 and 0..15.
    if a >= 345.0 or a < 15.0:
        return "Straight drive"

    for lo, hi, name in _SHOT_TABLE_RIGHT:
        if lo <= a < hi:
            return name

    return "Unknown"


def classify_shot_learned(
    *_args, **_kwargs,
) -> str:
    """
    Placeholder hook for a future learned classifier (e.g. LSTM head over
    the post-contact ground trajectory). Wire your model here and swap the
    call site in `analyse_delivery` if/when you train one.
    """
    raise NotImplementedError(
        "Learned shot classifier not implemented. "
        "Plug an LSTM / MLP here and call from analyse_delivery()."
    )


# ---------------------------------------------------------------------------
# 6. End-to-end wrapper
# ---------------------------------------------------------------------------

def analyse_delivery(
    ball_track: Dict[int, Tuple[float, float]],
    bat_boxes: Dict[int, Tuple[int, int, int, int]],
    H: np.ndarray,
    fps: float = 30.0,
    handedness: str = "right",
    *,
    proximity_px: float = 30.0,
    angle_change_deg: float = 25.0,
    contact_window: int = 3,
    post_contact_frames: int = 60,
    direction_lookahead: int = 10,
    interp_max_gap: int = 3,
    prediction_horizon_s: float = 2.0,
    prediction_step_s: float = 0.05,
    batsman_xy: Tuple[float, float] = (0.0, 0.0),
) -> ShotResult:
    """
    Run the full shot-analysis pipeline on a single delivery.

    Parameters
    ----------
    ball_track : dict
        frame_idx -> (x_px, y_px). The ball pixel track produced by the
        existing detector + tracker.
    bat_boxes : dict
        frame_idx -> (x1, y1, x2, y2). Bat detection bboxes per frame.
    H : (3,3) ndarray
        Pixel->ground homography from `calibrate_from_clicks`.
    fps : float
        Video frame rate.
    handedness : "right" | "left"
        Batsman handedness for shot-name lookup.

    Other kwargs are pipeline knobs documented in the README.

    Returns
    -------
    ShotResult
    """
    result = ShotResult(contact=None, handedness=handedness)

    if not ball_track:
        result.reason = "empty_ball_track"
        return result

    # 1. Contact
    contact = detect_contact(
        ball_track, bat_boxes,
        proximity_px=proximity_px,
        angle_change_deg=angle_change_deg,
        window=contact_window,
        interp_max_gap=interp_max_gap,
    )
    if contact is None:
        result.reason = "no_contact_detected"
        return result
    result.contact = contact

    # 2. Ground trajectory + prediction
    observed, predicted = fit_and_predict_ground(
        ball_track, H, contact.frame_idx, fps,
        post_contact_frames=post_contact_frames,
        prediction_horizon_s=prediction_horizon_s,
        prediction_step_s=prediction_step_s,
        interp_max_gap=interp_max_gap,
    )
    if len(observed) < 5:
        result.reason = "not_enough_post_contact_track"
        result.observed_ground = observed
        return result
    result.observed_ground = observed
    result.predicted_ground = predicted

    # 3. Angle + 4. shot name
    angle = compute_shot_angle(
        observed, contact.frame_idx,
        direction_lookahead=direction_lookahead,
        batsman_xy=batsman_xy,
    )
    if angle is None:
        result.reason = "could_not_compute_angle"
        return result
    result.angle_deg = angle
    result.shot_name = classify_shot(angle, handedness)
    return result


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------

def _draw_text(
    img: np.ndarray, text: str, org: Tuple[int, int],
    color=(0, 255, 255), scale: float = 0.7, thickness: int = 2,
) -> None:
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def render_overlay(
    input_video: str,
    output_video: str,
    ball_track: Dict[int, Tuple[float, float]],
    bat_boxes: Dict[int, Tuple[int, int, int, int]],
    result: ShotResult,
    H: np.ndarray,
) -> str:
    """
    Re-render `input_video` to `output_video` with overlays:
      * ball + bat bboxes (per frame)
      * vertical "CONTACT" marker on the contact frame
      * predicted 2-second ground trajectory re-projected to image space
      * angle (deg) and shot-name text

    Returns the output path on success.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Pre-project the predicted ground trajectory back to image space once.
    predicted_image_pts: Optional[np.ndarray] = None
    if result.predicted_ground:
        ground_pts = np.array(
            [(x, y) for (_t, x, y) in result.predicted_ground],
            dtype=np.float64,
        )
        predicted_image_pts = ground_to_image(ground_pts, H).astype(int)

    contact_frame = result.contact.frame_idx if result.contact else None
    interp_track = interpolate_track(ball_track, max_gap=3)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Bat box
            if frame_idx in bat_boxes:
                x1, y1, x2, y2 = bat_boxes[frame_idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                _draw_text(frame, "bat", (x1, max(0, y1 - 6)),
                           color=(255, 200, 0), scale=0.5, thickness=1)

            # Ball position
            if frame_idx in interp_track:
                bx, by = interp_track[frame_idx]
                cv2.circle(frame, (int(bx), int(by)), 6, (0, 0, 255), -1)

            # Predicted trajectory (only show after contact)
            if (
                predicted_image_pts is not None
                and contact_frame is not None
                and frame_idx >= contact_frame
            ):
                for i in range(1, len(predicted_image_pts)):
                    p0 = tuple(predicted_image_pts[i - 1])
                    p1 = tuple(predicted_image_pts[i])
                    if (
                        0 <= p0[0] < w and 0 <= p0[1] < h
                        and 0 <= p1[0] < w and 0 <= p1[1] < h
                    ):
                        cv2.line(frame, p0, p1, (0, 255, 0), 2)

            # Contact marker
            if contact_frame is not None and frame_idx == contact_frame:
                cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
                _draw_text(frame, "CONTACT", (20, 80),
                           color=(0, 255, 255), scale=1.0, thickness=2)

            # Persistent labels (after contact)
            if (
                contact_frame is not None
                and frame_idx >= contact_frame
                and result.angle_deg is not None
            ):
                _draw_text(
                    frame, f"Angle: {result.angle_deg:.1f} deg",
                    (20, 30), color=(0, 255, 255),
                )
                _draw_text(
                    frame, f"Shot:  {result.shot_name}",
                    (20, 55), color=(0, 255, 255),
                )

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_video


# ---------------------------------------------------------------------------
# Convenience: run detector+tracker over a clip to build the inputs
# ---------------------------------------------------------------------------

def build_tracks_from_video(
    video_path: str,
    ball_model_path: str,
    bat_model_path: Optional[str] = None,
    *,
    ball_class_id: int = 0,
    bat_class_id: int = 0,
    ball_conf: float = 0.2,
    bat_conf: float = 0.25,
) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[int, int, int, int]],
    float,
]:
    """
    Convenience helper that runs the existing YOLO ball detector + tracker
    over a clip and, optionally, a separate bat-detection YOLO model.

    If `bat_model_path` is None or the same as the ball model with multiple
    classes, you can pass `bat_class_id` to pick the bat class.

    Returns
    -------
    ball_track : dict
    bat_boxes  : dict
    fps        : float
    """
    # Local imports to keep top-level cheap and avoid mandatory ultralytics dep
    # for users who only want to call analyse_delivery() on pre-built tracks.
    from src.detection.yolo_detector import YoloBallDetector
    from src.tracking.tracker import BallTracker
    from src.association.data_association import associate_ball
    from ultralytics import YOLO  # noqa: F401

    detector = YoloBallDetector(
        ball_model_path, conf=ball_conf, ball_class_id=ball_class_id,
    )

    bat_model = None
    if bat_model_path:
        from ultralytics import YOLO as _YOLO
        bat_model = _YOLO(bat_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = BallTracker(fps=int(fps) or 30)

    ball_track: Dict[int, Tuple[float, float]] = {}
    bat_boxes: Dict[int, Tuple[int, int, int, int]] = {}

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Ball
            predicted = tracker.predict() if tracker.initialized else None
            dets = detector.detect(frame, predicted)
            match = associate_ball(dets, predicted)
            if match is not None:
                cx, cy = match[0], match[1]
                tracker.update((cx, cy))
                ball_track[frame_idx] = (float(cx), float(cy))

            # Bat
            if bat_model is not None:
                r = bat_model(frame, conf=bat_conf, verbose=False)[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    best_box = None
                    best_conf = 0.0
                    for box, cls, conf in zip(
                        r.boxes.xyxy, r.boxes.cls, r.boxes.conf,
                    ):
                        if int(cls) != bat_class_id:
                            continue
                        c = float(conf)
                        if c > best_conf:
                            best_conf = c
                            best_box = tuple(int(v) for v in box.tolist())
                    if best_box is not None:
                        bat_boxes[frame_idx] = best_box

            frame_idx += 1
    finally:
        cap.release()

    return ball_track, bat_boxes, fps


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny synthetic self-test: a ball travelling diagonally, deflecting off
    # the bat. Verifies that the contact detector, angle math and
    # classifier all work end-to-end without needing a video file.
    import pprint

    track = {}
    # Pre-contact: ball moving toward batsman, frames 0..14
    for f in range(15):
        track[f] = (500.0 + f * -5.0, 200.0 + f * 8.0)
    # Contact at frame 15 -> deflection to the off side
    for f in range(15, 60):
        track[f] = (track[14][0] + (f - 14) * 7.0,
                    track[14][1] + (f - 14) * -2.0)

    bx = track[15][0]
    by = track[15][1]
    bat = {f: (int(bx - 30), int(by - 60), int(bx + 30), int(by + 60))
           for f in range(10, 20)}

    # Trivial calibration: identity-ish (1 px == 0.02 m).
    img_pts = np.array(
        [[600, 100], [400, 100], [400, 500], [600, 500]], dtype=np.float64,
    )
    H = compute_homography(img_pts)

    res = analyse_delivery(track, bat, H, fps=30.0, handedness="right")
    pprint.pprint(res.to_dict())
