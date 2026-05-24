import json

class BallJSONExporter:
    def __init__(self):
        self.deliveries = []
        self.current_ball = None
        self.ball_id = 0

    def start_ball(self):
        self.ball_id += 1
        self.current_ball = {
            "ball_id": self.ball_id,
            "release": None,
            "release_speed_kmph": None,
            "bounce": None,
            "pitch_type": None,
            "trajectory": []
        }

    def add_position(self, frame_idx, x, y):
        if self.current_ball:
            self.current_ball["trajectory"].append({
                "frame": frame_idx,
                "x": int(x),
                "y": int(y)
            })

    def set_release(self, frame_idx, x, y):
        self.current_ball["release"] = {
            "frame": frame_idx,
            "x": int(x),
            "y": int(y)
        }

    def set_speed(self, speed):
        self.current_ball["release_speed_kmph"] = round(speed, 2)

    def set_bounce(self, x, y):
        self.current_ball["bounce"] = {
            "x": int(x),
            "y": int(y)
        }

    def set_pitch(self, pitch_type):
        self.current_ball["pitch_type"] = pitch_type

    def end_ball(self):
        if self.current_ball:
            self.deliveries.append(self.current_ball)
            self.current_ball = None

    def save(self, path="balls.json"):
        with open(path, "w") as f:
            json.dump(self.deliveries, f, indent=2)
