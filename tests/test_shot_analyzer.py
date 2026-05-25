import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot_analyzer import analyse_delivery, classify_shot, compute_homography


def _synthetic_delivery():
    track = {}
    for f in range(15):
        track[f] = (500.0 - f * 5.0, 200.0 + f * 8.0)
    for f in range(15, 60):
        track[f] = (track[14][0] + (f - 14) * 7.0, track[14][1] - (f - 14) * 2.0)

    bx, by = track[15]
    bat_boxes = {
        f: (int(bx - 30), int(by - 60), int(bx + 30), int(by + 60))
        for f in range(10, 20)
    }
    image_points = np.array(
        [[600, 100], [400, 100], [400, 500], [600, 500]],
        dtype=np.float64,
    )
    H = compute_homography(image_points)
    return track, bat_boxes, H


def test_analyse_delivery_synthetic_contact_and_prediction():
    track, bat_boxes, H = _synthetic_delivery()

    result = analyse_delivery(track, bat_boxes, H, fps=30.0, handedness="right")

    assert result.reason is None
    assert result.contact is not None
    assert result.contact.frame_idx == 13
    assert result.angle_deg is not None
    assert result.shot_name == "Straight drive"
    assert len(result.observed_ground) >= 5
    assert len(result.predicted_ground) == 41


def test_analyse_delivery_reports_no_contact():
    track, _bat_boxes, H = _synthetic_delivery()

    result = analyse_delivery(track, {}, H, fps=30.0)

    assert result.contact is None
    assert result.reason == "no_contact_detected"


def test_rule_based_shot_lookup_and_left_hand_mirror():
    assert classify_shot(0.0, "right") == "Straight drive"
    assert classify_shot(58.0, "right") == "Cover drive"
    assert classify_shot(270.0, "right") == "Pull"
    assert classify_shot(90.0, "left") == "Pull"
