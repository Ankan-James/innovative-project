
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
