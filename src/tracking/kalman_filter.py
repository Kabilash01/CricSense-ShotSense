import numpy as np

class BallKalmanFilter:
    """
    Kalman Filter for cricket ball tracking
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, dt=1/30):
        self.dt = dt

        # -------- State --------
        # x = [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # -------- State Transition --------
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # -------- Measurement Model --------
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # -------- Covariances --------
        self.P = np.eye(4) * 500        # Initial uncertainty
        self.R = np.eye(2) * 10         # Measurement noise
        self.Q = np.eye(4) * 0.1        # Process noise

        self.initialized = False

    # ------------------------------------
    def init(self, x, y):
        """Initialize filter with first detection"""
        self.x = np.array([[x], [y], [0], [0]])
        self.initialized = True

    # ------------------------------------
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    # ------------------------------------
    def update(self, z):
        """
        z: measurement [x, y]
        """
        z = np.array(z).reshape(2, 1)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    # ------------------------------------
    def get_position(self):
        return self.x[0, 0], self.x[1, 0]

    def get_velocity(self):
        return self.x[2, 0], self.x[3, 0]
