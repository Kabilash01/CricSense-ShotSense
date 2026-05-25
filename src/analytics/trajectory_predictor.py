import math
import numpy as np


class TrajectoryPredictor:

    def __init__(self,
                 fps=30,
                 friction=0.985,
                 predict_seconds=2.0):

        self.fps = fps
        self.dt = 1.0 / fps

        self.friction = friction
        self.predict_frames = int(fps * predict_seconds)

    # ---------------------------------------------------
    # Predict future trajectory from current state
    # ---------------------------------------------------
    def predict(self, x, y, vx, vy):

        future_points = []

        px = float(x)
        py = float(y)

        pvx = float(vx)
        pvy = float(vy)

        for _ in range(self.predict_frames):

            # next position
            px += pvx * self.dt
            py += pvy * self.dt

            # apply friction decay
            pvx *= self.friction
            pvy *= self.friction

            future_points.append((int(px), int(py)))

        return future_points

    # ---------------------------------------------------
    # Compute wagon-wheel angle
    # ---------------------------------------------------
    def compute_angle(self, origin, final_point):

        ox, oy = origin
        fx, fy = final_point

        dx = fx - ox
        dy = oy - fy

        angle = math.degrees(math.atan2(dx, dy))

        return round(angle, 2)

    # ---------------------------------------------------
    # Shot naming from angle
    # ---------------------------------------------------
    def classify_shot(self, angle):

        if -15 <= angle <= 15:
            return "Straight Drive"

        elif 15 < angle <= 45:
            return "Cover Drive"

        elif 45 < angle <= 75:
            return "Square Drive"

        elif 75 < angle <= 110:
            return "Cut Shot"

        elif -45 <= angle < -15:
            return "Flick"

        elif -75 <= angle < -45:
            return "Pull Shot"

        else:
            return "Sweep"

    # ---------------------------------------------------
    # Convert pixel distance → metres
    # ---------------------------------------------------
    def trajectory_distance_m(self,
                              points,
                              meters_per_pixel):

        if len(points) < 2:
            return 0.0

        total_px = 0.0

        for i in range(1, len(points)):

            x1, y1 = points[i - 1]
            x2, y2 = points[i]

            d = math.hypot(x2 - x1, y2 - y1)
            total_px += d

        return round(total_px * meters_per_pixel, 2)