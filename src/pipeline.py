
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

from src.camera.frame_sampler import FrameSampler
from src.vision.hand_detector import HandDetector
from src.vision.hand_cropper import HandCropper
from src.inference.vit_classifier import ViTClassifier
from src.inference.temporal_voting import TemporalVoting


ROOT = Path(__file__).resolve().parents[1]


class VideoSignInterpreter:

    def __init__(self):
        print("Initializing SLMCI...")

        self.detector = HandDetector()
        self.cropper = HandCropper(padding=30)
        self.classifier = ViTClassifier()

        self.voting = TemporalVoting(
            window_size=8,
            min_votes=5,
            min_confidence=0.50
        )

        self.stable_sign = None

        print("SLMCI initialized successfully!")

    def predict_frame(self, frame):
        """
        Detect a hand and classify its sign.

        Returns:
            label, confidence, bounding_box
        """

        hands = self.detector.detect(frame)

        if not hands:
            return None, 0.0, None

        hand = hands[0]
        box = hand["box"]

        cropped = self.cropper.crop(frame, hand)

        if cropped is None or cropped.size == 0:
            return None, 0.0, None

        # Convert OpenCV BGR to PIL RGB
        rgb = cv2.cvtColor(
            cropped,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        # Use the same transforms as training
        tensor = self.classifier.transform(image)
        tensor = tensor.unsqueeze(0).to(
            self.classifier.model.device
        )

        with torch.inference_mode():
            outputs = self.classifier.model(
                pixel_values=tensor
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

            confidence, predicted_id = torch.max(
                probabilities,
                dim=0
            )

        label = self.classifier.model.config.id2label[
            predicted_id.item()
        ]

        return label, confidence.item(), box

    def process_video(self, video_path):

        video_path = Path(video_path)

        if not video_path.is_absolute():
            video_path = ROOT / video_path

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open video. Check its format."
            )

        source_fps = cap.get(cv2.CAP_PROP_FPS)

        if source_fps <= 0:
            source_fps = 24

        sampler = FrameSampler(
            source_fps=source_fps,
            target_fps=8
        )

        frame_index = 0
        analyzed_frames = 0

        current_label = "Waiting..."
        current_confidence = 0.0
        current_box = None

        print("\nProcessing:", video_path.name)
        print("Video FPS:", source_fps)
        print("Analysis FPS:", sampler.target_fps)
        print("Press Q to stop.\n")

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                if sampler.should_sample(frame_index):

                    analyzed_frames += 1

                    label, confidence, box = (
                        self.predict_frame(frame)
                    )

                    current_box = box
                    current_confidence = confidence

                    if label is None:
                        self.voting.reset()
                        self.stable_sign = None
                        current_label = "No hand"

                    else:
                        current_label = label

                        result = self.voting.update(
                            label,
                            confidence
                        )

                        if result is not None:
                            if result != self.stable_sign:
                                print(
                                    f"Stable sign: {result}"
                                )

                            self.stable_sign = result

                # Draw hand bounding box
                if current_box is not None:
                    x1, y1, x2, y2 = current_box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # Display current frame prediction
                cv2.putText(
                    frame,
                    f"Frame: {current_label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {current_confidence * 100:.1f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                stable_text = (
                    self.stable_sign
                    if self.stable_sign
                    else "Analyzing..."
                )

                cv2.putText(
                    frame,
                    f"Stable sign: {stable_text}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    "SLMCI - Video Interpreter",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_index += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("\nVideo processing complete!")
        print("Total video frames:", frame_index)
        print("Frames analyzed:", analyzed_frames)

    def close(self):
        self.detector.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.pipeline "
            "data\\recordings\\sign_test.mp4"
        )
        sys.exit(1)

    interpreter = VideoSignInterpreter()

    try:
        interpreter.process_video(sys.argv[1])

    finally:
        interpreter.close()
