import numpy as np

def associate_ball(detections, predicted_pos, max_dist=120):
    if not detections:
        return None

    if predicted_pos is None:
        return detections[0]

    px, py = predicted_pos
    best, best_dist = None, float("inf")

    for det in detections:
        cx, cy = det[0], det[1]
        dist = np.hypot(cx - px, cy - py)

        if dist < best_dist and dist < max_dist:
            best = det
            best_dist = dist

    return best
