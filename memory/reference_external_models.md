---
name: External model sources and reference repos
description: Where the project's pretrained weights and reference architectures come from
type: reference
---

- **Shot classifier weights** (EfficientNetB0 + GRU, 10-class): https://github.com/RITIK-12/CricketShotClassification — `model_weights.h5` downloaded by `scripts/convert_to_onnx.py`. Classes: cover, defense, flick, hook, late_cut, lofted, pull, square_cut, straight, sweep.
- **Detectron2** (planned for Ubuntu): https://github.com/facebookresearch/detectron2 — install from prebuilt wheels matching local CUDA version. Use Keypoint R-CNN config for 17 COCO keypoints.
- **Ball detector**: trained in-house, weights at `ball_test/weights/best.pt` (YOLOv8, classes: 0=ball, 1=stump)
- **Bat detector**: trained in-house, weights at `models/bat_detector_v8n/weights/best.pt` (YOLOv8, classes: 0='-', 1='bat' — note the unused class 0)
