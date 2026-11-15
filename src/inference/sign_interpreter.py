
class SignInterpreter:
    """
    Converts stable ASL predictions into text.
    Prevents the same held sign from being added repeatedly.
    """

    LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")

    def __init__(self):
        self.text = ""
        self.last_sign = None
        self.no_hand_count = 0

    def update(self, label, hand_detected=True):
        """
        Called once for each completed 8-frame prediction.

        label:
            A-Y (excluding J and Z), SPACE, DEL,
            NOTHING, or None.
        """

        # Only clear the held-sign state when the
        # hand has been deliberately absent.
        if label == "NOTHING" or (
            label is None and not hand_detected
        ):
            self.no_hand_count += 1

            if self.no_hand_count >= 1:
                self.last_sign = None

            return self.get_result(False)

        # An inconclusive prediction is not a new sign.
        if label is None:
            self.no_hand_count = 0
            return self.get_result(False)

        self.no_hand_count = 0

        # Ignore unsupported labels.
        if label not in self.LETTERS | {"SPACE", "DEL"}:
            return self.get_result(False)

        # Don't repeat a sign that is being held.
        if label == self.last_sign:
            return self.get_result(False)

        # Handle special signs.
        if label == "SPACE":
            if self.text and not self.text.endswith(" "):
                self.text += " "

        elif label == "DEL":
            self.text = self.text[:-1]

        else:
            # Add the recognized alphabet.
            self.text += label

        self.last_sign = label

        return self.get_result(True)

    def get_result(self, accepted):
        return {
            "accepted": accepted,
            "text": self.text,
            "current_word": self.text.split(" ")[-1]
        }

    def clear(self):
        self.text = ""
        self.last_sign = None
        self.no_hand_count = 0


# ----------------------------------
# TEST WITHOUT WEBCAM
# ----------------------------------

if __name__ == "__main__":

    interpreter = SignInterpreter()

    signs = ["A", "N", "K", "A", "N"]

    print("Testing Sign Interpreter...\n")

    for sign in signs:
        result = interpreter.update(sign)

        print(
            f"Sign: {sign} | "
            f"Formed word: {result['current_word']}"
        )

    print("\nFinal word:", interpreter.text)
