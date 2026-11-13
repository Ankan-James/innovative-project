
from collections import deque, Counter


class TemporalVoting:
    def __init__(
        self,
        window_size=8,
        min_votes=5,
        min_confidence=0.50
    ):
        if window_size < 1:
            raise ValueError("Window size must be positive")

        if not 1 <= min_votes <= window_size:
            raise ValueError("Invalid minimum vote count")

        self.window_size = window_size
        self.min_votes = min_votes
        self.min_confidence = min_confidence

        # Store recent frame predictions
        self.history = deque(maxlen=window_size)

    def update(self, label, confidence=1.0):
        """
        Add a prediction from one video frame.

        Returns the stable predicted letter,
        or None if no stable prediction exists.
        """

        # Clear old predictions when the hand disappears
        if label is None or label == "NOTHING":
            self.reset()
            return None

        # Ignore uncertain predictions
        if confidence < self.min_confidence:
            return None

        self.history.append(label)

        # Count how often each sign appears
        counts = Counter(self.history)

        best_label, votes = counts.most_common(1)[0]

        # Require enough agreement between frames
        if votes >= self.min_votes:
            return best_label

        return None

    def reset(self):
        """Clear prediction history."""
        self.history.clear()

    def get_history(self):
        """Return recent predictions."""
        return list(self.history)


# ---------------------------------
# TEST WITHOUT A WEBCAM
# ---------------------------------

if __name__ == "__main__":

    voting = TemporalVoting(
        window_size=8,
        min_votes=5,
        min_confidence=0.50
    )

    # Simulated predictions from 8 video frames
    predictions = [
        "A", "A", "S", "A",
        "A", "E", "A", "A"
    ]

    print("Testing temporal voting...\n")

    for i, label in enumerate(predictions, start=1):

        result = voting.update(
            label=label,
            confidence=0.95
        )

        print(
            f"Frame {i}: {label} "
            f"| Stable: {result}"
        )

    print("\nPrediction history:", voting.get_history())
    print("Final predicted sign:", result)
