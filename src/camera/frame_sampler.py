from pathlib import Path
import cv2
from PIL import Image

class FrameSampler:
    def __init__(self, source_fps=24, target_fps=8):
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError("FPS must be greater than zero")

        self.source_fps = source_fps
        self.target_fps = min(target_fps, source_fps)

    def should_sample(self, frame_index):
        """
        Returns True when a frame should be analyzed.
        """
        if frame_index < 0:
            raise ValueError("Frame index cannot be negative")

        if frame_index == 0:
            return True

        current = int(
            frame_index * self.target_fps / self.source_fps
        )

        previous = int(
            (frame_index - 1) * self.target_fps / self.source_fps
        )

        return current > previous
    
    
def sample_video(video_path, num_frames=8):
    """
    Extract exactly 8 evenly spaced frames from a video.
    Returns a list of PIL Images.
    """

    if num_frames <= 0:
        raise ValueError("Number of frames must be positive")

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            "Cannot open video. Check the video format or codec."
        )

    frames = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frames.append(frame)

            # Prevent unexpectedly long uploads
            # from consuming too much memory.
            if len(frames) > 120:
                raise ValueError(
                    "Video is too long. Maximum 120 decoded frames."
                )

    finally:
        capture.release()

    total_frames = len(frames)

    if total_frames < num_frames:
        raise ValueError(
            f"Video has only {total_frames} frames; "
            f"at least {num_frames} are required."
        )

    # Select evenly spaced frames across the whole clip.
    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = [
            round(i * (total_frames - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]

    sampled_frames = []

    for index in indices:
        frame = frames[index]

        # OpenCV uses BGR; MediaPipe/ViT need RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        sampled_frames.append(
            Image.fromarray(rgb_frame)
        )

    print(f"Total video frames: {total_frames}")
    print(f"Selected frame indices: {indices}")

    return sampled_frames


if __name__ == "__main__":
    sampler = FrameSampler(
        source_fps=24,
        target_fps=8
    )

    selected_frames = []

    for frame_index in range(24):
        if sampler.should_sample(frame_index):
            selected_frames.append(frame_index)

    print("Original FPS:", sampler.source_fps)
    print("Analysis FPS:", sampler.target_fps)
    print("Selected frame indices:", selected_frames)
    print("Total selected frames:", len(selected_frames))
