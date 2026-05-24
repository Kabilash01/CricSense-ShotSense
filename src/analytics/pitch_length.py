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

