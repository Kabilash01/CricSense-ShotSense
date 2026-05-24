from collections import deque
from ultralytics import YOLO
import numpy as np
import cv2
import time
import os

# ================= CONFIG =================
MODEL_PATH = r"C:\cricket player train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
BALL_CLASS_ID = 0        # change if your ball class id is different
CONF_THRESH = 0.1
FPS = 30
# ==========================================


# ========== SIMPLE KALMAN FILTER ==========
class BallKalman:
    def __init__(self, dt=1/30):
        self.dt = dt
        self.x = np.zeros((4, 1))   # [x, y, vx, vy]
        self.P = np.eye(4) * 500

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.Q = np.eye(4) * 0.5
        self.R = np.eye(2) * 10

        self.initialized = False
        self.prev_vy = 0

    def init(self, x, y):
        self.x = np.array([[x], [y], [0], [0]])
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()

# ==========================================


# -------- Load model --------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

kalman = BallKalman(dt=1/FPS)
trajectory = deque(maxlen=20)

prev_time = time.time()

# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- FPS ----------
    curr_time = time.time()
    fps = int(1 / max(curr_time - prev_time, 1e-6))
    prev_time = curr_time

    # ---------- YOLO DETECTION ----------
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    detection = None
    best_conf = 0

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != BALL_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if conf > best_conf:
                best_conf = conf
                detection = (cx, cy, x1, y1, x2, y2)

    # ---------- KALMAN UPDATE ----------
    if detection:
        cx, cy, x1, y1, x2, y2 = detection

        if not kalman.initialized:
            kalman.init(cx, cy)
        else:
            kalman.predict()
            kalman.update((cx, cy))

        # draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    elif kalman.initialized:
        kalman.predict()

    # ---------- TRACKING OUTPUT ----------
    if kalman.initialized:
        x, y, vx, vy = kalman.get_state()
        x, y = int(x), int(y)

        trajectory.append((x, y))

        # draw tracked ball
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)

        # draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (255,0,0), 2)

        # ---------- BOUNCE DETECTION ----------
        if kalman.prev_vy < 0 and vy > 0:
            cv2.putText(frame, "BALL BOUNCED",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        kalman.prev_vy = vy

        # ---------- FUTURE PREDICTION ----------
        future = []
        for i in range(1, 6):
            fx = int(x + vx * i)
            fy = int(y + vy * i)
            future.append((fx, fy))

        for i in range(1, len(future)):
            cv2.line(frame, future[i-1], future[i], (0,255,0), 2)
            cv2.circle(frame, future[i], 3, (0,255,0), -1)

    # ---------- DISPLAY ----------
    cv2.putText(frame, f'FPS: {fps}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    frame_resized = cv2.resize(frame, (1000,600))
    cv2.imshow("Ball Tracking (YOLO + Kalman)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
cap.release()
cv2.destroyAllWindows()
