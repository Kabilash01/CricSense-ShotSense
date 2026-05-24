import math
from collections import deque

class ReleaseSpeedEstimator:
    def __init__(self, fps, window=8):
        self.fps = fps
        self.window = window
        self.positions = deque(maxlen=window + 1)

        self.released = False
        self.release_speed = None
        self.trigger_count = 0

    def update(self, pos, speed_kmph):
        if self.release_speed is not None:
            return self.release_speed

        # detect release
        if speed_kmph > 30:   # release threshold
            self.trigger_count += 1
        else:
            self.trigger_count = 0

        if self.trigger_count >= 3 and not self.released:
            self.released = True
            self.positions.clear()

        if self.released:
            self.positions.append(pos)

            if len(self.positions) >= self.window:
                self.compute_speed()

        return self.release_speed

    def compute_speed(self):
        (x0, y0) = self.positions[0]
        (x1, y1) = self.positions[-1]

        dist_px = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        time_sec = len(self.positions) / self.fps

        self.release_speed = dist_px / time_sec
