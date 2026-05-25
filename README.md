# CricSense Ball Tracking and Trajectory

Python/OpenCV cricket ball tracking with YOLO detection, tracking, release speed estimation, and a new shot-analysis pipeline.

The original detector and tracker under `src/` are left untouched. Shot analysis lives in one notebook-friendly module: [`shot_analyzer.py`](shot_analyzer.py).

## Shot Analysis Pipeline

The new pipeline takes:

- `ball_track`: `frame_idx -> (x_px, y_px)` from the existing YOLO ball detector and tracker
- `bat_boxes`: `frame_idx -> (x1, y1, x2, y2)` from a bat YOLO detector
- `H`: a pixel-to-ground homography from one-time pitch calibration
- `fps` and batsman handedness

It returns:

- Frame where bat met ball
- Observed post-contact ground trajectory in metres
- 2-second predicted ground trajectory in metres
- Wagon-wheel shot angle in degrees
- Rule-based shot name, such as Straight drive, Cover drive, Square cut, Pull
- Optional annotated output video

Pipeline stages:

```text
existing YOLO ball detector + BallTracker
        -> ball_track and optional ball_boxes

bat YOLO detector
        -> bat_boxes

analyse_delivery()
        -> interpolate short ball-track gaps
        -> detect bat-ball contact using proximity + velocity angle change
        -> project post-contact ball points to ground metres with H
        -> fit degree-2 X(t), Y(t) trajectory
        -> predict 0-2 seconds ahead
        -> compute wagon-wheel angle
        -> classify shot by angle table

render_overlay()
        -> annotated MP4 with ball/bat boxes, CONTACT marker, trajectory, label
```

## Quick Start

```python
from shot_analyzer import (
    build_tracks_from_video,
    calibrate_from_clicks,
    load_homography,
    analyse_delivery,
    render_overlay,
)

VIDEO_PATH = "data/samples/test_video.mp4"
BALL_MODEL_PATH = "ball_test/weights/best.pt"
BAT_MODEL_PATH = "path/to/bat_yolo.pt"

ball_track, ball_boxes, bat_boxes, fps = build_tracks_from_video(
    VIDEO_PATH,
    ball_model_path=BALL_MODEL_PATH,
    bat_model_path=BAT_MODEL_PATH,
    ball_class_id=0,
    bat_class_id=0,
    include_ball_boxes=True,
)

# Run this once per camera setup, then reuse the saved JSON.
H = calibrate_from_clicks(VIDEO_PATH, save_path="outputs/calibration_H.json")
# H = load_homography("outputs/calibration_H.json")

result = analyse_delivery(
    ball_track,
    bat_boxes,
    H,
    fps=fps,
    handedness="right",
)

print(result.to_dict())

render_overlay(
    VIDEO_PATH,
    "outputs/videos/shot_overlay.mp4",
    ball_track,
    bat_boxes,
    result,
    H,
    ball_boxes=ball_boxes,
)
```

See [`demo.ipynb`](demo.ipynb) for a notebook walkthrough.

## Calibration

Cricket pitch dimensions used by the homography:

- Length: 22 yards = 20.12 m
- Width: 10 ft = 3.05 m

Ground coordinate frame:

- Origin: striker's wicket centre
- `+Y`: toward the bowler
- `+X`: off side for a right-handed batsman

Run:

```python
from shot_analyzer import calibrate_from_clicks

H = calibrate_from_clicks(
    "data/samples/test_video.mp4",
    save_path="outputs/calibration_H.json",
)
```

Click the first video frame in this fixed order:

1. Striker-end off-side pitch corner
2. Striker-end leg-side pitch corner
3. Bowler-end leg-side pitch corner
4. Bowler-end off-side pitch corner

Press `r` to reset clicks, `q` to abort, or any key after 4 clicks to confirm.

Use `load_homography("outputs/calibration_H.json")` on later runs. Recalibrate only when the camera moves.

## Contact Detection

`detect_contact()` uses two cues and both must agree:

- Ball centre is within `proximity_px` of the bat bbox.
- Ball velocity angle changes by at least `angle_change_deg`, comparing the average vector over `contact_window` frames before and after the candidate frame.

Before contact detection, the module linearly interpolates short ball-track gaps up to `interp_max_gap` frames. This helps with motion blur when YOLO misses the exact contact frame.

The returned `ContactEvent` contains:

- `frame_idx`
- `ball_xy`
- `bat_box`
- `angle_change_deg`
- `proximity_px`
- `confidence`

## Shot Angle

The wagon-wheel angle is measured clockwise from `+Y`, which means straight back toward the bowler is 0 degrees.

```text
0 deg   = straight drive
90 deg  = square cut / off side
180 deg = behind keeper
270 deg = pull / leg side
```

The direction vector is computed from the batsman origin to the ground position of the ball about `direction_lookahead` frames after contact.

## Shot Classification

Right-handed lookup table:

| Angle range | Shot |
| --- | --- |
| 345-15 | Straight drive |
| 15-40 | Off drive |
| 40-75 | Cover drive |
| 75-105 | Square cut |
| 105-150 | Late cut / Upper cut |
| 150-200 | Edge (slip / keeper area) |
| 200-250 | Leg glance / Sweep |
| 250-285 | Pull |
| 285-320 | Mid-wicket pull / Flick |
| 320-345 | On drive |

For a left-handed batsman, the module mirrors the angle with `(360 - angle) % 360` before applying the same table.

`classify_shot_learned()` is a deliberate placeholder for a future learned LSTM/MLP head. The v1 path is rule-based.

## Tuneable Knobs

All are exposed as `analyse_delivery()` keyword arguments:

| Knob | Default | Meaning |
| --- | --- | --- |
| `proximity_px` | `30` | Max ball-centre distance from bat bbox for contact |
| `angle_change_deg` | `25` | Min velocity direction change across contact |
| `contact_window` | `3` | Frames before/after candidate contact used for velocity |
| `post_contact_frames` | `60` | Post-contact frames used for trajectory fitting |
| `direction_lookahead` | `10` | Frames after contact used for shot direction |
| `interp_max_gap` | `3` | Interpolate ball-track gaps up to this many frames |
| `prediction_horizon_s` | `2.0` | Future prediction horizon |
| `prediction_step_s` | `0.05` | Prediction sample interval |
| `batsman_xy` | `(0, 0)` | Ground origin used for angle computation |

## Edge Cases

`analyse_delivery()` returns a JSON-safe result with an informative `reason` instead of crashing:

- `empty_ball_track`
- `no_contact_detected`
- `not_enough_post_contact_track`
- `could_not_compute_angle`

## Sample Output

```json
{
  "contact": {
    "frame_idx": 47,
    "ball_xy": [612.0, 405.0],
    "bat_box": [590, 370, 640, 460],
    "angle_change_deg": 67.4,
    "proximity_px": 12.3,
    "confidence": 0.87
  },
  "observed_ground": [
    {"frame": 47, "x_m": 0.12, "y_m": 0.05}
  ],
  "predicted_ground": [
    {"t_s": 0.0, "x_m": 0.12, "y_m": 0.05},
    {"t_s": 0.05, "x_m": 0.31, "y_m": 0.18}
  ],
  "angle_deg": 58.7,
  "shot_name": "Cover drive",
  "handedness": "right",
  "reason": null
}
```

## Requirements

Install with:

```bash
python -m pip install -r requirements.txt
```

Core dependencies:

- NumPy
- OpenCV
- PyTorch
- Ultralytics
- pytest

## Self Test

Run the synthetic end-to-end module test:

```bash
python shot_analyzer.py
```

This does not need a model or a video. It creates a synthetic delivery, detects contact, projects to ground coordinates, predicts the 2-second trajectory, computes the angle, and classifies the shot.
