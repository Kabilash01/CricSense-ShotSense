import json

class BallJSONExporter:
    def __init__(self, video_name, fps, pitch_length=20.12):
        self.data = {
            "video_id": video_name,
            "fps": fps,
            "pitch_length_meters": pitch_length,
            "deliveries": []
        }
        self.current_delivery = None
        self.delivery_id = 0

    def start_delivery(self, start_frame):
        self.delivery_id += 1
        self.current_delivery = {
            "delivery_id": self.delivery_id,
            "timestamps": {
                "start_frame": start_frame,
                "release_frame": None,
                "bounce_frame": None,
                "end_frame": None
            },
            "release": {},
            "bounce": {},
            "speed": {
                "release_kmph": None,
                "max_kmph": 0.0,
                "average_kmph": 0.0
            },
            "pitch": {},
            "trajectory": {
                "path_px": []
            }
        }

    def add_position(self, x, y):
        if self.current_delivery:
            self.current_delivery["trajectory"]["path_px"].append(
                {"x": int(x), "y": int(y)}
            )

    def set_release(self, frame_id, x, y, speed):
        self.current_delivery["timestamps"]["release_frame"] = frame_id
        self.current_delivery["release"] = {
            "frame": frame_id,
            "position_px": {"x": int(x), "y": int(y)},
            "speed_kmph": round(speed, 2)
        }
        self.current_delivery["speed"]["release_kmph"] = round(speed, 2)

    def set_bounce(self, frame_id, x, y):
        self.current_delivery["timestamps"]["bounce_frame"] = frame_id
        self.current_delivery["bounce"] = {
            "frame": frame_id,
            "position_px": {"x": int(x), "y": int(y)}
        }

    def update_speed(self, speed):
        self.current_delivery["speed"]["max_kmph"] = max(
            self.current_delivery["speed"]["max_kmph"], speed
        )

    def finalize_delivery(self, end_frame, pitch_type):
        self.current_delivery["timestamps"]["end_frame"] = end_frame
        self.current_delivery["pitch"]["type"] = pitch_type
        self.current_delivery["trajectory"]["num_points"] = len(
            self.current_delivery["trajectory"]["path_px"]
        )
        self.data["deliveries"].append(self.current_delivery)
        self.current_delivery = None

    def save(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=4)
