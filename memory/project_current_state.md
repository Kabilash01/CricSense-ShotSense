---
name: Cricket-Angle pipeline current state (2026-05-30)
description: Where test3.py stands, what works, what's broken, and the planned next step
type: project
---

`src/test3.py` is the active pipeline. As of 2026-05-30:

**Working**:
- Ball detection: YOLOv8 at `ball_test/weights/best.pt` (conf=0.10) — sparse, ~3 hits per 300 frames
- Bat detection: YOLOv8 at `models/bat_detector_v8n/weights/best.pt` (conf=0.05, bat_class_id=1, classes are `{0:'-', 1:'bat'}`) — detects many false positives (stumps, umpire, helmet)
- Shot classifier: EfficientNetB0+GRU ONNX at `models/shot_classifier/shot_classifier.onnx` — converted from RITIK-12/CricketShotClassification h5 weights via `scripts/convert_to_onnx.py`. Loads on CUDAExecutionProvider.
- Contact detection: bat-box proximity (primary) + trajectory curvature/accel-spike (fallback)
- Phantom-speed guard: resets tracker if speed > 800 px/frame
- 45-frame post-contact window — analysis now runs ONCE at window close (fixed 2026-05-30)

**Recently fixed (2026-05-30)**:
- `Events=0 despite contacts firing` bug: analyze-shot block used to run every frame as soon as `post_contact_points > 4`. Within 4 frames the post-contact points were all stale identical tracker predictions, so the zero-movement guard killed the window before real ball motion accumulated. Fix: moved analysis into the `else` branch that fires when `frames_since_contact >= 45`, added `if tracker_updated:` gate on post-contact appends, and added a duplicate-point filter before smoothing.

**Known remaining issues**:
- Bat detector misclassifies stumps/umpire/helmet as bats — user explicitly flagged this
- Ball detection too sparse for reliable trajectory-based contact triggering
- User reports "missing bat-ball contact sometimes" even with current fix

**Why**: User wants reliable per-delivery event extraction for analytics (`events.json`).

**How to apply**:
- Default to pose-based triggering (Detectron2 on Ubuntu) as the next architectural step — see `project_next_step_detectron2.md`
- Don't propose more tweaks to the curvature/accel thresholds — that path is exhausted
- When verifying current behavior, re-read `src/test3.py` before quoting line numbers (file changes rapidly)
