class BoundaryDetector:
    def __init__(self, boundary_y_ratio=0.92):
        """
        boundary_y_ratio:
        0.92 = bottom 8% of frame is boundary area
        """
        self.boundary_y_ratio = boundary_y_ratio
        self.triggered = False

    def reset(self):
        self.triggered = False

    def detect(self, trajectory, frame_height, has_bounced):
        """
        trajectory: list of (x, y)
        frame_height: frame.shape[0]
        has_bounced: bool
        """
        if self.triggered or len(trajectory) < 3:
            return None

        boundary_y = int(frame_height * self.boundary_y_ratio)

        # check last movement
        prev_y = trajectory[-2][1]
        curr_y = trajectory[-1][1]

        # crossing boundary line
        if prev_y < boundary_y and curr_y >= boundary_y:
            self.triggered = True
            if has_bounced:
                return "FOUR"
            else:
                return "SIX"

        return None
