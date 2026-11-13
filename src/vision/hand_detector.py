
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import sys


# Project root
ROOT = Path(__file__).resolve().parents[2]

# MediaPipe model
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


class HandDetector:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {MODEL_PATH}"
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3
        )

        self.detector = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        )

    def detect(self, frame):
        """
        Detect hands in an OpenCV BGR image.

        Returns:
            list of hand dictionaries containing
            landmarks and bounding boxes.
        """
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )

        result = self.detector.detect(image)

        hands = []

        for landmarks in result.hand_landmarks:
            points = []

            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]

            box = (
                max(0, min(xs) - 25),
                max(0, min(ys) - 25),
                min(width, max(xs) + 25),
                min(height, max(ys) + 25)
            )

            hands.append({
                "landmarks": landmarks,
                "points": points,
                "box": box
            })

        return hands

    def draw(self, frame, hands):
        """Draw detected landmarks and bounding boxes."""
        output = frame.copy()

        for hand in hands:
            x1, y1, x2, y2 = hand["box"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            for x, y in hand["points"]:
                cv2.circle(
                    output,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        return output

    def close(self):
        self.detector.close()


# ---------------------------------------
# TEST WITH AN IMAGE (NO WEBCAM REQUIRED)
# ---------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.vision.hand_detector "
            "path_to_image.jpg"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    detector = HandDetector()

    try:
        hands = detector.detect(frame)

        print("Hands detected:", len(hands))

        result = detector.draw(frame, hands)

        cv2.imshow("SLMCI - Hand Detection", result)
        cv2.waitKey(0)

    finally:
        detector.close()
        cv2.destroyAllWindows()
