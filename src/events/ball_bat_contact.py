import math

class BallBatContact:
    def __init__(self,
                 dist_thresh=25,
                 speed_drop_thresh=0.15,
                 angle_change_thresh=25):
        self.prev_speed = None
        self.prev_angle = None
        self.contact_fired = False

        self.dist_thresh = dist_thresh
        self.speed_drop_thresh = speed_drop_thresh
        self.angle_change_thresh = angle_change_thresh

    def reset(self):
        self.prev_speed = None
        self.prev_angle = None
        self.contact_fired = False

    def detect(self, ball_pos, ball_vel, bat_box):
        """
        Returns: (contact: bool, confidence: float, contact_type: str)
        """
        if bat_box is None or self.contact_fired:
            return False, 0.0, None

        bx, by = ball_pos
        vx, vy = ball_vel

        # --- distance check ---
        x1, y1, x2, y2 = bat_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        dist = math.hypot(bx - cx, by - cy)
        if dist > self.dist_thresh:
            return False, 0.0, None

        # --- speed & angle ---
        speed = math.hypot(vx, vy)
        angle = math.degrees(math.atan2(vy, vx + 1e-6))

        if self.prev_speed is None:
            self.prev_speed = speed
            self.prev_angle = angle
            return False, 0.0, None

        speed_drop = (self.prev_speed - speed) / max(self.prev_speed, 1e-6)
        angle_change = abs(angle - self.prev_angle)

        self.prev_speed = speed
        self.prev_angle = angle

        if speed_drop > self.speed_drop_thresh or angle_change > self.angle_change_thresh:
            self.contact_fired = True
            contact_type = self._classify_hit(bx, by, bat_box)
            confidence = min(1.0, speed_drop + angle_change / 90)
            return True, confidence, contact_type

        return False, 0.0, None

    def _classify_hit(self, bx, by, bat_box):
        x1, y1, x2, y2 = bat_box
        h = y2 - y1
        rel_y = (by - y1) / max(h, 1)

        if rel_y < 0.25:
            return "top_edge"
        elif rel_y > 0.75:
            return "bottom_edge"
        else:
            return "middle"
