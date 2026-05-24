# src/events/ball_bat.py
import math
from collections import deque

class BallBatContact:
    def __init__(self, angle_thresh=25, speed_ratio=1.25):
        self.angle_thresh = angle_thresh
        self.speed_ratio = speed_ratio
        self.prev_v = None
        self.triggered = False

    def reset(self):
        self.prev_v = None
        self.triggered = False

    def detect(self, vx, vy):
        """
        vx, vy: current velocity (px/sec)
        returns: (contact_detected: bool, confidence: float)
        """
        if self.triggered:
            return False, 0.0

        speed = math.hypot(vx, vy)

        if self.prev_v is None:
            self.prev_v = (vx, vy, speed)
            return False, 0.0

        pvx, pvy, pspeed = self.prev_v

        # Angle change
        dot = pvx * vx + pvy * vy
        mag1 = math.hypot(pvx, pvy)
        mag2 = math.hypot(vx, vy)

        if mag1 == 0 or mag2 == 0:
            self.prev_v = (vx, vy, speed)
            return False, 0.0

        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = math.degrees(math.acos(cos_theta))

        # Speed change
        speed_ratio = speed / (pspeed + 1e-6)

        # Contact condition
        if angle > self.angle_thresh and speed_ratio > self.speed_ratio:
            self.triggered = True
            confidence = min(1.0, (angle / 90.0) * (speed_ratio / 2.0))
            return True, confidence

        self.prev_v = (vx, vy, speed)
        return False, 0.0
