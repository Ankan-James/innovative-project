
import cv2
import sys
from pathlib import Path

from src.vision.hand_detector import HandDetector


ROOT = Path(__file__).resolve().parents[2]


class HandCropper:
    def __init__(self, padding=30):
        self.padding = padding

    def crop(self, image, hand):
        """
        Extract the detected hand from an image.
        """
        height, width = image.shape[:2]

        x1, y1, x2, y2 = hand["box"]

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(width, x2 + self.padding)
        y2 = min(height, y2 + self.padding)

        if x2 <= x1 or y2 <= y1:
            return None

        cropped_hand = image[y1:y2, x1:x2]

        return cropped_hand.copy()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python -m src.vision.hand_cropper A1.jpg")
        sys.exit(1)

    image_path = ROOT / sys.argv[1]

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector = HandDetector()
    cropper = HandCropper()

    try:
        hands = detector.detect(image)

        print(f"Hands detected: {len(hands)}")

        output_dir = ROOT / "data" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, hand in enumerate(hands):
            cropped = cropper.crop(image, hand)

            if cropped is None:
                continue

            output_path = output_dir / f"hand_crop_{i + 1}.jpg"

            cv2.imwrite(str(output_path), cropped)

            print(f"Hand crop saved: {output_path}")

        print("Hand cropping completed!")

    finally:
        detector.close()
