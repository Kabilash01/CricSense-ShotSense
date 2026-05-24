import math
from collections import deque

class BallTracker:
    def __init__(self, fps=30, window=6, ema_alpha=0.25):
        self.fps = fps
        self.dt = 1.0 / fps

        self.positions = deque(maxlen=window)
        self.trajectory = []

        self.initialized = False

        self.vx = 0.0
        self.vy = 0.0

        self.raw_speed = 0.0
        self.display_speed = 0.0
        self.max_speed = 0.0
        self.prev_raw_speed = None

        self.ema_alpha = ema_alpha

        self.release_point = None
        self.release_speed = None

        self.has_bounced = False
        self.bounce_y = None
        self.pitch_type = None

        self.missed_frames = 0

    def reset(self):
        self.__init__(self.fps)

    def update(self, pos):
        self.positions.append(pos)
        self.trajectory.append(pos)

        if not self.initialized:
            self.release_point = pos
            self.initialized = True
            return

        if len(self.positions) < 2:
            return

        (x1, y1) = self.positions[0]
        (x2, y2) = self.positions[-1]
        frames = len(self.positions) - 1

        self.vx = (x2 - x1) * self.fps / frames
        self.vy = (y2 - y1) * self.fps / frames

    def predict(self):
        if not self.initialized or not self.positions:
            return None
        x, y = self.positions[-1]
        return int(x + self.vx * self.dt), int(y + self.vy * self.dt)

    def get_position(self):
        return self.positions[-1]

    def get_speed_kmph(self, meters_per_pixel, scale=1.0):
        speed_px = math.sqrt(self.vx**2 + self.vy**2)
        raw = speed_px * meters_per_pixel * 3.6 * scale

        if self.prev_raw_speed is not None:
            max_delta = 18.0
            raw = max(self.prev_raw_speed - max_delta,
                      min(raw, self.prev_raw_speed + max_delta))

        self.prev_raw_speed = raw
        self.raw_speed = raw

        if self.display_speed == 0.0:
            self.display_speed = raw
        else:
            self.display_speed = (
                self.ema_alpha * raw +
                (1 - self.ema_alpha) * self.display_speed
            )

        if self.release_speed is None and self.display_speed > 10:
            self.release_speed = self.display_speed

        self.max_speed = max(self.max_speed, self.display_speed)
        return self.display_speed

    def detect_bounce(self):
        if not self.initialized or len(self.positions) < 4:
            return False

        (_, y1) = self.positions[-3]
        (_, y2) = self.positions[-2]
        (_, y3) = self.positions[-1]

        if not self.has_bounced and y2 > y1 and y3 < y2:
            self.has_bounced = True
            self.bounce_y = y2
            return True

        return False

    def classify_pitch(self, frame_height):
        if self.bounce_y is None:
            return None

        y_norm = self.bounce_y / frame_height

        if y_norm > 0.75:
            return "YORKER"
        elif y_norm > 0.55:
            return "FULL"
        elif y_norm > 0.35:
            return "GOOD"
        else:
            return "SHORT"
        
    def get_velocity(self):
        return self.vx, self.vy