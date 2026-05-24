# Cricket-Angle / CricSense — Shot Analysis Pipeline

Real-time cricket ball tracking + shot analysis from video.

The existing repo handles ball detection (YOLO), tracking (Kalman), bounce
detection, pitch-length classification and release-speed estimation. This
update adds a **shot analysis** layer on top of that:

* **Bat-ball contact detection** (proximity + velocity-angle fusion, with
  short-gap interpolation to survive motion blur)
* **Ground-plane homography** (4-click calibration on the pitch)
* **Post-contact ground trajectory + 1-2 s prediction** (degree-2 polyfit on
  ground X/Y)
* **Shot angle** (wagon-wheel angle in degrees, clockwise from "toward bowler")
* **Rule-based shot classification** (Straight drive / Cover drive / Square cut
  / Pull / ...) with right/left-hand handedness support
* **Video overlay** of bat & ball bboxes, contact marker, predicted trajectory,
  and shot label
* **Clean hook for a learned LSTM classifier** later

Everything lives in a single importable module: [shot_analyzer.py](shot_analyzer.py).
A runnable walkthrough is in [demo.ipynb](demo.ipynb).

---

## Pipeline overview

```text
  YOLO ball detector  ──┐
                        ├─►  ball_track (frame -> px)
  BallTracker (Kalman) ─┘                    │
                                             ▼
  YOLO bat detector  ──►  bat_boxes (frame -> bbox)
                                             │
                                             ▼
                              ┌──── analyse_delivery() ────┐
                              │                            │
                              │  1) interpolate gaps       │
                              │  2) detect contact         │
                              │  3) ground-project + fit   │
                              │  4) compute shot angle     │
                              │  5) classify shot          │
                              └────────────┬───────────────┘
                                           │
                                           ▼
                       ShotResult { contact, observed_ground,
                                    predicted_ground, angle, shot_name }
                                           │
                                           ▼
                              render_overlay(...) → mp4
```

---

## Quick start

```python
from shot_analyzer import (
    build_tracks_from_video, calibrate_from_clicks,
    analyse_delivery, render_overlay,
)

# 1) Run the existing detector + tracker over the clip.
ball_track, bat_boxes, fps = build_tracks_from_video(
    video_path="data/samples/test_video.mp4",
    ball_model_path="models/yolo/ball_stump_best.pt",
    bat_model_path="path/to/bat_yolo.pt",      # or None
)

# 2) Calibrate the ground plane once (or load a saved one).
H = calibrate_from_clicks(
    "data/samples/test_video.mp4",
    save_path="outputs/calibration_H.json",
)

# 3) Run the analysis.
result = analyse_delivery(
    ball_track, bat_boxes, H,
    fps=fps, handedness="right",
)
print(result.to_dict())

# 4) Render the annotated video.
render_overlay(
    "data/samples/test_video.mp4",
    "outputs/videos/shot_overlay.mp4",
    ball_track, bat_boxes, result, H,
)
```

See [demo.ipynb](demo.ipynb) for the same flow in a notebook, plus a top-down
wagon-wheel plot of the ground trajectory.

---

## Calibration

Cricket pitch dimensions are fixed by the laws of the game:

* Length: 22 yards = **20.12 m**
* Width:  10 ft    = **3.05 m**

The ground reference frame used by this pipeline:

* **Origin** = striker's wicket centre
* **+Y** = toward the bowler (down the pitch)
* **+X** = off side (for a right-handed batsman)

`calibrate_from_clicks(video_path)` opens the first frame and asks you to click
the 4 pitch corners in this order:

1. Striker-end **off**-side corner
2. Striker-end **leg**-side corner
3. Bowler-end **leg**-side corner
4. Bowler-end **off**-side corner

It then computes the homography `H` (pixels → metres) via
`cv2.findHomography`, saves it to JSON, and returns it. Use `load_homography()`
on subsequent runs.

You only need to recalibrate if the camera moves.

---

## Shot-angle convention

Angle is measured **clockwise from +Y** (i.e. "straight toward the bowler"),
in the range `[0°, 360°)`, computed from the batsman position to the ball
position roughly 10 frames after contact (all in ground metres).

For a right-handed batsman:

| Angle range | Shot |
| --- | --- |
| 345°–15° | Straight drive |
| 15°–40° | Off drive |
| 40°–75° | Cover drive |
| 75°–105° | Square cut |
| 105°–150° | Late cut / Upper cut |
| 150°–200° | Edge (slip / keeper area) |
| 200°–250° | Leg glance / Sweep |
| 250°–285° | Pull |
| 285°–320° | Mid-wicket pull / Flick |
| 320°–345° | On drive |

For a left-handed batsman the angle is mirrored — internally we compute
`(360 - angle) % 360` before looking up the table.

A learned classifier head can be plugged in via `classify_shot_learned(...)`;
the rule-based version is the v1 default.

---

## Tuneable knobs

All exposed as kwargs to `analyse_delivery()`:

| Knob | Default | Meaning |
| --- | --- | --- |
| `proximity_px` | `30` | Max distance (px) from ball centre to bat bbox for contact |
| `angle_change_deg` | `25` | Min velocity-angle change (deg) across contact frame |
| `contact_window` | `3` | Frames used to compute pre/post-contact velocity |
| `post_contact_frames` | `60` | Frames sampled after contact for trajectory fit |
| `direction_lookahead` | `10` | Frames after contact used to estimate shot direction |
| `interp_max_gap` | `3` | Linear-interp ball track over gaps ≤ this size |
| `prediction_horizon_s` | `2.0` | Future trajectory horizon |
| `prediction_step_s` | `0.05` | Future trajectory sample step |
| `batsman_xy` | `(0,0)` | Origin used for the angle computation |

---

## Edge cases

`analyse_delivery` returns a populated `reason` field instead of crashing:

* `empty_ball_track` — track dict was empty.
* `no_contact_detected` — neither cue, or only one cue, ever fired.
* `not_enough_post_contact_track` — fewer than 5 valid frames after the
  contact frame (cannot fit a stable polynomial).
* `could_not_compute_angle` — degenerate direction vector.

---

## Files added / changed

| File | Role |
| --- | --- |
| [shot_analyzer.py](shot_analyzer.py) | The full module (homography, contact, trajectory, angle, classification, overlay, calibration, end-to-end wrapper) |
| [demo.ipynb](demo.ipynb) | Runnable notebook walkthrough on a sample clip |
| [README.md](README.md) | This file |
| [CLAUDE.md](CLAUDE.md) | Repo context for AI assistants |

The existing detector / tracker under [src/](src/) is **unchanged** — the new
module just imports from it.

---

## Sample output

`result.to_dict()` returns a JSON-safe payload like:

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
    {"frame": 47, "x_m": 0.12, "y_m": 0.05},
    ...
  ],
  "predicted_ground": [
    {"t_s": 0.00, "x_m": 0.12, "y_m": 0.05},
    {"t_s": 0.05, "x_m": 0.31, "y_m": 0.18},
    ...
    {"t_s": 2.00, "x_m": 8.42, "y_m": 4.81}
  ],
  "angle_deg": 58.7,
  "shot_name": "Cover drive",
  "handedness": "right",
  "reason": null
}
```

---

## Requirements

Same as the base repo — no new heavy deps:

```text
numpy>=1.24.0
opencv-python>=4.8.0
ultralytics>=8.0.0
```

`matplotlib` is optional, only for the wagon-wheel plot cell in the notebook.

---

## Self-test

Running the module directly executes a synthetic end-to-end test
(no video required):

```bash
python shot_analyzer.py
```

It builds a fake delivery, runs contact detection, ground projection, the
2-second prediction, angle computation and shot classification, and prints
the JSON-safe result.
