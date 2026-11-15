"""Experimental ASL J/Z fingertip motion tracker and webcam preview."""

from collections import deque
import time

import cv2
import numpy as np

from src.vision.hand_detector import HandDetector


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

BLUE = (255, 0, 0)       # OpenCV uses BGR
PINK = (203, 75, 255)
GREEN = (0, 210, 0)


def finger_is_raised(landmarks, mcp, pip, tip):
    """Roughly determine if a finger is straight using its PIP joint."""
    pts = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
    a = pts[mcp] - pts[pip]
    b = pts[tip] - pts[pip]
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return False

    angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1)))
    tip_distance = np.linalg.norm(pts[tip] - pts[mcp])
    joint_distance = np.linalg.norm(pts[pip] - pts[mcp])
    return angle > 150 and tip_distance > 1.45 * joint_distance


def get_active_finger(landmarks):
    """Only track when exactly the index OR the pinky is raised."""
    fingers = {
        "INDEX": finger_is_raised(landmarks, 5, 6, 8),
        "MIDDLE": finger_is_raised(landmarks, 9, 10, 12),
        "RING": finger_is_raised(landmarks, 13, 14, 16),
        "PINKY": finger_is_raised(landmarks, 17, 18, 20),
    }
    raised = [name for name, enabled in fingers.items() if enabled]
    if raised == ["INDEX"]:
        return "INDEX"
    if raised == ["PINKY"]:
        return "PINKY"
    return None


class MotionClassifier:
    """Experimental rule-based detector; requires a single recorded gesture."""

    def __init__(self, min_frames=12):
        self.min_frames = min_frames

    @staticmethod
    def _smooth(path):
        if len(path) < 3:
            return path
        padded = np.pad(path, ((1, 1), (0, 0)), mode="edge")
        return (padded[:-2] + padded[1:-1] + padded[2:]) / 3

    @staticmethod
    def _looks_like_z(path):
        # Three alternating strokes: horizontal, diagonal down, horizontal.
        n = len(path)
        for i in range(2, n - 4):
            for j in range(i + 2, n - 1):
                a = path[i] - path[0]
                b = path[j] - path[i]
                c = path[-1] - path[j]
                if not (abs(a[0]) > 0.35 and abs(b[0]) > 0.35 and abs(c[0]) > 0.35):
                    continue
                if not (a[0] * b[0] < 0 and b[0] * c[0] < 0):
                    continue
                if (abs(a[1]) < 0.5 * abs(a[0]) + 0.15
                        and b[1] > 0.25
                        and abs(c[1]) < 0.5 * abs(c[0]) + 0.15):
                    return True
        return False

    @staticmethod
    def _looks_like_j(path):
        # Downward movement with a sideways and upward hook at the end.
        bottom = int(np.argmax(path[:, 1]))
        if bottom < len(path) // 4 or bottom >= len(path) - 2:
            return False
        down = path[bottom, 1] - path[0, 1]
        up = path[bottom, 1] - path[-1, 1]
        sideways = abs(path[-1, 0] - path[bottom, 0])
        return down > 0.4 and up > 0.12 and sideways > 0.2

    def predict(self, landmark_frames, active_finger):
        if len(landmark_frames) < self.min_frames:
            return None
        pts = np.asarray(landmark_frames, dtype=np.float32)
        if pts.ndim != 3 or pts.shape[1:] != (21, 3):
            raise ValueError("Expected frames containing 21 (x, y, z) landmarks")

        palm_sizes = np.linalg.norm(pts[:, 9, :2] - pts[:, 0, :2], axis=1)
        scale = max(float(np.median(palm_sizes)), 1e-6)

        if active_finger == "INDEX":
            path = self._smooth((pts[:, 8, :2] - pts[0, 8, :2]) / scale)
            return "Z" if self._looks_like_z(path) else None
        if active_finger == "PINKY":
            path = self._smooth((pts[:, 20, :2] - pts[0, 20, :2]) / scale)
            return "J" if self._looks_like_j(path) else None
        return None


def draw_trail(image, points, color):
    if len(points) >= 2:
        cv2.polylines(
            image, [np.asarray(points, dtype=np.int32)],
            isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA
        )


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam. Close other apps using it.")

    detector = HandDetector()
    classifier = MotionClassifier()
    index_trail = deque(maxlen=70)
    pinky_trail = deque(maxlen=70)
    landmarks_sequence = []
    recording = False
    record_start = 0.0
    record_finger = None
    last_prediction = "Not tested"
    duration = 2.0

    print("SLMCI Motion Tracker: R = record, C = clear, Q = quit")
    print("Raise ONLY your index (blue) or pinky (pink).")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            hands = detector.detect(frame)
            display = detector.draw(frame, hands)
            active = None

            if hands:
                hand = hands[0]
                pts = hand["points"]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(display, pts[start], pts[end], GREEN, 2)

                active = get_active_finger(hand["landmarks"])
                if active == "INDEX":
                    pinky_trail.clear()
                    index_trail.append(pts[8])
                    cv2.circle(display, pts[8], 7, BLUE, -1)
                elif active == "PINKY":
                    index_trail.clear()
                    pinky_trail.append(pts[20])
                    cv2.circle(display, pts[20], 7, PINK, -1)
                else:
                    index_trail.clear()
                    pinky_trail.clear()

                if recording and active is not None:
                    if record_finger is None:
                        record_finger = active
                    if active == record_finger:
                        landmarks_sequence.append([
                            (p.x, p.y, p.z) for p in hand["landmarks"]
                        ])
            else:
                index_trail.clear()
                pinky_trail.clear()

            draw_trail(display, index_trail, BLUE)
            draw_trail(display, pinky_trail, PINK)

            if recording:
                remaining = max(0, duration - (time.perf_counter() - record_start))
                status = f"RECORDING: {remaining:.1f}s"
                if remaining <= 0:
                    recording = False
                    result = classifier.predict(landmarks_sequence, record_finger)
                    last_prediction = result or "None"
                    print(f"Tracked: {record_finger}, Frames: {len(landmarks_sequence)}, Prediction: {last_prediction}")
            else:
                status = "Press R to record J or Z"

            preview = cv2.flip(display, 1)
            cv2.putText(preview, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            cv2.putText(preview, f"Active finger: {active or 'NONE'}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            cv2.putText(preview, f"Prediction: {last_prediction}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("SLMCI - J/Z Motion Tracker", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                index_trail.clear()
                pinky_trail.clear()
                landmarks_sequence = []
                record_finger = None
                record_start = time.perf_counter()
                last_prediction = "Recording..."
                recording = True
                print("Recording started")
            elif key == ord("c"):
                index_trail.clear()
                pinky_trail.clear()
            elif key == ord("q"):
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
