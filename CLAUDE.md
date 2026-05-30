
# Cricket-Angle — Project Context

## Active Work (as of 2026-05-30)

The current pipeline is `src/test3.py` (not `test.py` / `test2.py` — those are older). It uses:

- YOLOv8 ball detector: `ball_test/weights/best.pt` (conf=0.10, class 0=ball)
- YOLOv8 bat detector: `models/bat_detector_v8n/weights/best.pt` (conf=0.05, **bat_class_id=1** because model classes are `{0:'-', 1:'bat'}`)
- ONNX EfficientNetB0+GRU shot classifier: `models/shot_classifier/shot_classifier.onnx` (converted from RITIK-12/CricketShotClassification via `scripts/convert_to_onnx.py`)

**Open problem**: bat-ball contacts are missed (ball detection is sparse, bat detector hallucinates on stumps/umpire/helmet).

**Agreed next step**: add Detectron2 pose-based wrist-velocity triggering as a third parallel contact trigger. Implement on Ubuntu, not Windows — Detectron2 on Windows is a known multi-hour install. See `memory/project_next_step_detectron2.md` for the full plan.

## Ubuntu Migration Quickstart

If you are reading this on Ubuntu after a Git pull:

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Install Detectron2 from prebuilt wheel matching local CUDA version (see [Detectron2 install guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html))
4. Update absolute paths in `src/test3.py` CONFIG block — change `C:\Cricket-Angle\...` to repo-relative paths
5. Read `memory/MEMORY.md` for full context before making decisions
6. Verify ONNX classifier loads: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` — should list `CUDAExecutionProvider`

## Project Goal

Real-time cricket ball tracking and analytics system. Processes cricket match video to extract per-delivery telemetry: bowling speed, ball trajectory, bounce detection, pitch zone classification, boundary events (FOUR/SIX), and bat-contact events. Outputs annotated video + structured JSON.

## Tech Stack

- **Python 3.x**
- **YOLOv8** (`ultralytics>=8.0.0`) — ball detection, boundary line detection, shot classification
- **OpenCV** (`opencv-python>=4.8.0`) — video I/O, frame processing, annotation
- **NumPy** (`numpy>=1.24.0`) — Kalman filter matrices, numerical ops
- **PyTorch** — GPU acceleration (commented out in requirements, needed for YOLO)
- **pytest** (`pytest>=7.0.0`) — unit testing

## Directory Structure

```
Cricket-Angle/
├── CLAUDE.md
├── requirements.txt
├── README.md (empty)
│
├── src/                        # Main source code
│   ├── main.py                 # Entry point: basic YOLO + Kalman demo
│   ├── ball_tracking.py        # YOLO + Kalman + trajectory visualization
│   ├── test.py                 # Full pipeline: detection + tracking + pitch classification
│   ├── test2.py                # Full pipeline + boundary detection + shot classification
│   │
│   ├── detection/
│   │   ├── yolo_detector.py    # YoloBallDetector — detects ball per frame
│   │   ├── boundary_detector.py # BoundaryLineDetector — detects field boundary
│   │   └── shot_classifier.py  # ShotClassifier — classifies shot type
│   │
│   ├── tracking/
│   │   ├── kalman_filter.py    # BallKalmanFilter — [x,y,vx,vy] state estimation
│   │   └── tracker.py          # BallTracker — position history, speed EMA, release point
│   │
│   ├── association/
│   │   └── data_association.py # associate_ball() — nearest-neighbor detection matching
│   │
│   ├── events/
│   │   ├── ball_bat.py / ball_bat_contact.py  # Bat-contact detection + hit type
│   │   ├── boundary.py / boundary_event.py    # FOUR/SIX classification
│   │   └── boundary_logic.py                  # intersects() geometry helper
│   │
│   ├── analytics/
│   │   ├── release_speed.py    # ReleaseSpeedEstimator
│   │   ├── speed_estimator.py  # Speed calculation helpers
│   │   ├── bounce_detector.py  # (stub)
│   │   └── pitch_length.py     # (stub)
│   │
│   ├── export/
│   │   └── ball_json.py        # BallJSONExporter — per-ball JSON serialization
│   │
│   ├── utils/
│   │   ├── geometry.py         # (stub)
│   │   ├── json_exporter.py    # JSON output
│   │   └── logger.py           # (stub)
│   │
│   ├── trajectory/
│   │   └── trajectory_builder.py # (stub — logic lives in BallTracker)
│   │
│   └── visualization/
│       └── draw.py             # (stub — drawing done inline in test scripts)
│
├── models/
│   └── yolo/
│       └── ball_stump_best.pt  # PRIMARY YOLO model (ball + stump detection)
│
├── ball_test/
│   └── weights/
│       ├── best.pt             # Best checkpoint from ball_test training run
│       └── last.pt             # Last checkpoint from ball_test training run
│
├── data/
│   ├── raw/videos/match_01.mp4
│   └── samples/test_video.mp4
│
├── outputs/
│   ├── json/ball_trajectory.json
│   ├── plots/trajectory_plot.png
│   └── videos/tracked_output.mp4
│
├── configs/
│   ├── kalman.yaml             # (empty stub)
│   ├── tracker.yaml            # (empty stub)
│   └── yolo.yaml               # (empty stub)
│
├── scripts/
│   ├── train_yolo.py
│   ├── infer_video.py
│   ├── test_yolo_only.py
│   └── export_results.py
│
└── tests/
    ├── test_kalman.py
    └── test_tracker.py
```

## Available YOLO Checkpoints

| File | Path | Notes |
|---|---|---|
| `ball_stump_best.pt` | `models/weights/ball_stump_best.pt` | Empty placeholder — ignore |
| `best.pt` | `ball_test/weights/best.pt` | **YOLOv8 ball detector** — class 0=ball, class 1=stump |
| `best.pt` | `models/weights/best.pt` | **YOLOv5 bat detector** — class 0=ball, class 1=bat |
| `last.pt` | `ball_test/weights/last.pt` | YOLOv8 last epoch — usually worse than best |

> **Important:** The hardcoded paths in `src/main.py`, `src/ball_tracking.py`, and `src/test.py` point to `C:\cricket player train\...` and `C:\CricketSense-Ball\...` which do NOT exist. Always update these to `models/weights/ball_stump_best.pt` (relative) or the absolute path within this repo.
> **Note:** `models/yolo/` was renamed to `models/weights/` to eliminate a Python namespace collision with ultralytics internals during YOLOv5 pickle deserialization.

## Key Configuration Constants

```python
METERS_PER_PIXEL = 18.5 / 520   # Cricket pitch calibration (22 yards ≈ 18.5m over ~520px)
CONF_THRESH = 0.1 to 0.25       # YOLO confidence threshold
FPS = 30                         # Frame rate
BALL_CLASS_ID = 0               # YOLO class index for ball
```

## Processing Pipeline (Data Flow)

```
Video Input
  → Frame loop
  → YoloBallDetector.detect(frame)           # YOLO ball detection
  → tracker.predict()                         # Kalman prediction
  → associate_ball(detections, predicted)     # Nearest-neighbor matching (max 120px)
  → BallTracker.update(position)             # Position history, speed EMA
  → Event detection:
      ├─ bounce → pitch zone (YORKER/FULL/GOOD/SHORT)
      ├─ BallBatContact.detect()              # Bat hit + type
      └─ BoundaryDetector → FOUR/SIX
  → BallJSONExporter (per ball)
  → Annotated frame write → output video
  → ball_analysis.json, events.json
```

## Pitch Zone Classification (by bounce Y-coordinate)

| Zone | Y threshold (normalized) | Color |
|---|---|---|
| YORKER | y > 0.75 | Red |
| FULL | y > 0.55 | Yellow |
| GOOD | y > 0.35 | Green |
| SHORT | y < 0.35 | Blue |

## Speed Calculation

```python
speed_kmph = sqrt(vx² + vy²) × METERS_PER_PIXEL × perspective_scale(y, h) × 3.6
display_speed = EMA(alpha=0.4, raw_speed)    # Smoothed display value
```

Release is detected when speed > 30 km/h for 3+ consecutive frames.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Basic Kalman demo
python src/main.py

# Full tracking pipeline (recommended)
python src/test.py

# Full pipeline + boundary + shot classification
python src/test2.py

# Run tests
pytest tests/
```

> Before running, update `MODEL_PATH` and `VIDEO_PATH` in the script you're using to point to valid local files.

## Known Issues / Incomplete Parts

- `configs/*.yaml` files are all empty — config loading not implemented
- `src/visualization/draw.py` is a stub — drawing is done inline in test scripts
- `src/utils/geometry.py` and `src/utils/logger.py` are stubs
- `src/analytics/bounce_detector.py` and `pitch_length.py` are stubs
- `src/trajectory/trajectory_builder.py` is a stub (logic is in `BallTracker`)
- No boundary or shot classifier YOLO models are present in this repo (only ball model)
- Single-ball tracking only — no multi-object tracking
- METERS_PER_PIXEL is hardcoded and must be recalibrated per camera/pitch setup

## Output Files

| File | Contents |
|---|---|
| `ball_analysis.json` | Per-delivery: ball_id, release speed, max speed, pitch type, bounce y, release point |
| `events.json` | Discrete cricket events (boundaries, bat contacts) |
| `output.json` | Rich event logs with context |
| `*_output.mp4` | Annotated video with overlays |
