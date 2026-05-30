---
name: Next step — Detectron2 pose-based contact triggering on Ubuntu
description: Agreed architectural direction for replacing the broken ball+bat contact trigger
type: project
---

**Decision (2026-05-30)**: Move repo to Ubuntu via Git and add Detectron2 pose-based shot triggering as a parallel path to the current ball/bat triggers.

**Why**:
- User evaluated two research papers (optical flow vs Detectron2+XGBoost pose). Assistant recommended the pose approach because the current pipeline's bottleneck is the trigger, not the EfficientNet classifier — the ball detector is too sparse and the bat detector has too many false positives to reliably fire contact events.
- Detectron2 chosen over MediaPipe because: (a) higher keypoint accuracy on partial occlusion and bent-over stances (sweep, defensive); (b) Paper 2 from RITIK-12-adjacent literature uses Detectron2 specifically; (c) user has GPU and accepts the install cost on Ubuntu.
- Detectron2 on Windows is a known multi-hour install (specific CUDA + MSVC + pycocotools manual build) — explicitly why the migration to Ubuntu was chosen.

**How to apply when resuming on Ubuntu**:
1. Fresh clone, set up venv, `pip install -r requirements.txt` plus Detectron2 from facebookresearch wheels matching CUDA version
2. Add a new `src/detection/pose_detector.py` wrapping Detectron2 Keypoint R-CNN — return 17 COCO keypoints + per-keypoint confidence
3. In `src/test3.py`, add a third contact-trigger branch: wrist-velocity spike (compute frame-to-frame delta of wrist keypoints; threshold tuned empirically). Fires alongside bat-box and trajectory triggers.
4. Keep existing EfficientNet ONNX classifier — only the trigger changes, not the classifier
5. Update CLAUDE.md "Available YOLO Checkpoints" with Detectron2 model path once downloaded

**Do NOT**:
- Try Detectron2 on Windows (user explicitly moved off Windows to avoid this)
- Replace the EfficientNet shot classifier — it works once triggered
- Re-tune the curvature/accel thresholds — that path is exhausted
- Adopt the Paper 1 optical-flow approach — rejected because broadcast video has too much camera pan/zoom that dominates pixel-level flow
