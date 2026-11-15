"""
SLMCI — Sign Language Interpreter
Converts stable ASL predictions into text with an Adaptive Sign Correction System.
"""


class SignInterpreter:
    """
    Converts stable ASL predictions into text.
    Prevents the same held sign from being added repeatedly.
    Includes an Adaptive Sign Correction System for K/V and A/T confusion pairs.
    """

    LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")

    # Strictly limited confusion pairs
    CONFUSION_PAIRS = {
        "K": "V",
        "V": "K",
        "A": "T",
        "T": "A",
    }

    # Correction State Machine constants
    STATE_NORMAL = "NORMAL"
    STATE_WAITING_FOR_RETRY = "WAITING_FOR_RETRY"
    STATE_CORRECTION_APPLIED = "CORRECTION_APPLIED"

    def __init__(self):
        self.text = ""
        self.last_sign = None
        self.no_hand_count = 0

        # Adaptive Correction State Machine
        self.correction_state = self.STATE_NORMAL
        self.rejected_letter = None
        self.replacement_letter = None
        self.retry_index = None
        self.applied_letter = None
        self.applied_index = None
        self.suppressed_pair = None
        self.held_display_label = None

        # Last prediction metadata
        self.last_raw_label = None
        self.last_accepted_label = None
        self.last_adaptive_corrected = False

    def _reset_correction_state(self, keep_suppression=False):
        """Reset internal adaptive correction state variables."""
        self.correction_state = self.STATE_NORMAL
        self.rejected_letter = None
        self.replacement_letter = None
        self.retry_index = None
        self.applied_letter = None
        self.applied_index = None
        if not keep_suppression:
            self.suppressed_pair = None

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
                self.held_display_label = None

            return self.get_result(False, raw_label=label)

        # An inconclusive prediction is not a new sign.
        if label is None:
            self.no_hand_count = 0
            return self.get_result(False, raw_label=None)

        self.no_hand_count = 0

        # Ignore unsupported labels.
        if label not in self.LETTERS | {"SPACE", "DEL"}:
            return self.get_result(False, raw_label=label)

        # Don't repeat a sign that is being held.
        if label == self.last_sign:
            display_lbl = self.held_display_label or label
            return self.get_result(
                False,
                label=display_lbl,
                raw_label=label,
                accepted_label=self.held_display_label,
                adaptive_corrected=bool(self.held_display_label and self.held_display_label != label)
            )

        # Invalidate correction state if text length does not match expected position
        if self.correction_state == self.STATE_WAITING_FOR_RETRY:
            if len(self.text) != self.retry_index:
                self._reset_correction_state()
        elif self.correction_state == self.STATE_CORRECTION_APPLIED:
            if len(self.text) - 1 != self.applied_index or not self.text.endswith(self.applied_letter):
                self._reset_correction_state()

        # Handle special signs.
        if label == "SPACE":
            # Space commits text and cancels any pending retry / applied correction
            self._reset_correction_state()
            self.held_display_label = None

            if self.text and not self.text.endswith(" "):
                self.text += " "

            self.last_sign = "SPACE"
            return self.get_result(
                True,
                label="SPACE",
                raw_label="SPACE",
                accepted_label="SPACE",
                adaptive_corrected=False
            )

        elif label == "DEL":
            self.last_sign = "DEL"
            self.held_display_label = None

            if not self.text:
                # DEL on empty text does nothing; do not activate correction.
                return self.get_result(
                    True,
                    label="DEL",
                    raw_label="DEL",
                    accepted_label="DEL",
                    adaptive_corrected=False
                )

            deleted_char = self.text[-1]
            self.text = self.text[:-1]

            # Scenario 1: User deletes the character that was just applied by adaptive correction
            # (or kept as the accepted alternative in CORRECTION_APPLIED)
            if (
                self.correction_state == self.STATE_CORRECTION_APPLIED
                and len(self.text) == self.applied_index
                and deleted_char == self.applied_letter
            ):
                # Second DEL: revert and reset correction to NORMAL.
                # Crucial: do not immediately activate another correction for this pair.
                # The next signing attempt must use raw AI prediction.
                alt_pair = self.CONFUSION_PAIRS.get(deleted_char)
                suppressed = frozenset([deleted_char, alt_pair]) if alt_pair else None
                self._reset_correction_state()
                self.suppressed_pair = suppressed

            # Scenario 2: User deletes an eligible confusion letter in NORMAL state
            elif (
                self.correction_state == self.STATE_NORMAL
                and deleted_char in self.CONFUSION_PAIRS
                and (self.suppressed_pair is None or deleted_char not in self.suppressed_pair)
            ):
                self.correction_state = self.STATE_WAITING_FOR_RETRY
                self.rejected_letter = deleted_char
                self.replacement_letter = self.CONFUSION_PAIRS[deleted_char]
                self.retry_index = len(self.text)
                self.suppressed_pair = None

            # Scenario 3: Any other deletion
            else:
                if (
                    deleted_char in self.CONFUSION_PAIRS
                    and (self.suppressed_pair is None or deleted_char not in self.suppressed_pair)
                ):
                    self.correction_state = self.STATE_WAITING_FOR_RETRY
                    self.rejected_letter = deleted_char
                    self.replacement_letter = self.CONFUSION_PAIRS[deleted_char]
                    self.retry_index = len(self.text)
                    self.suppressed_pair = None
                else:
                    self._reset_correction_state()

            return self.get_result(
                True,
                label="DEL",
                raw_label="DEL",
                accepted_label="DEL",
                adaptive_corrected=False
            )

        else:
            raw_label = label
            accepted_label = label
            adaptive_corrected = False

            if self.correction_state == self.STATE_WAITING_FOR_RETRY:
                if label == self.rejected_letter:
                    # Model repeated the rejected confusion letter -> substitute paired alternative
                    accepted_label = self.replacement_letter
                    adaptive_corrected = True
                    self.text += accepted_label

                    self.correction_state = self.STATE_CORRECTION_APPLIED
                    self.applied_letter = accepted_label
                    self.applied_index = len(self.text) - 1
                    self.rejected_letter = None
                    self.replacement_letter = None
                    self.retry_index = None
                    self.suppressed_pair = None

                elif label == self.replacement_letter:
                    # Model directly predicted the paired alternative -> keep it
                    accepted_label = label
                    adaptive_corrected = False
                    self.text += accepted_label

                    self.correction_state = self.STATE_CORRECTION_APPLIED
                    self.applied_letter = accepted_label
                    self.applied_index = len(self.text) - 1
                    self.rejected_letter = None
                    self.replacement_letter = None
                    self.retry_index = None
                    self.suppressed_pair = None

                else:
                    # Unrelated letter -> accept normally and cancel pending correction
                    accepted_label = label
                    adaptive_corrected = False
                    self.text += accepted_label
                    self._reset_correction_state()

            elif self.correction_state == self.STATE_CORRECTION_APPLIED:
                # User continues typing normally after a correction was applied
                accepted_label = label
                adaptive_corrected = False
                self.text += accepted_label
                self._reset_correction_state()

            else:  # STATE_NORMAL
                accepted_label = label
                adaptive_corrected = False
                self.text += accepted_label
                # Clear one-shot suppression
                self.suppressed_pair = None

            self.last_sign = raw_label
            self.held_display_label = accepted_label if adaptive_corrected else None

            return self.get_result(
                True,
                label=accepted_label,
                raw_label=raw_label,
                accepted_label=accepted_label,
                adaptive_corrected=adaptive_corrected
            )

    def get_result(
        self,
        accepted,
        label=None,
        raw_label=None,
        accepted_label=None,
        adaptive_corrected=None
    ):
        if accepted:
            if label is not None:
                self.last_accepted_label = label
            if raw_label is not None:
                self.last_raw_label = raw_label
            if accepted_label is not None:
                self.last_accepted_label = accepted_label
            if adaptive_corrected is not None:
                self.last_adaptive_corrected = adaptive_corrected
        elif label is None and self.held_display_label:
            label = self.held_display_label

        effective_label = label if label is not None else (self.last_accepted_label if accepted else None)
        effective_raw = raw_label if raw_label is not None else (self.last_raw_label if accepted else None)
        effective_accepted = accepted_label if accepted_label is not None else (self.last_accepted_label if accepted else None)
        effective_adaptive = adaptive_corrected if adaptive_corrected is not None else (self.last_adaptive_corrected if accepted else False)

        return {
            "accepted": accepted,
            "text": self.text,
            "current_word": self.text.split(" ")[-1] if self.text else "",
            "label": effective_label,
            "raw_label": effective_raw,
            "accepted_label": effective_accepted,
            "adaptive_corrected": bool(effective_adaptive)
        }

    def clear(self):
        self.text = ""
        self.last_sign = None
        self.no_hand_count = 0
        self.held_display_label = None
        self._reset_correction_state(keep_suppression=False)
        self.last_raw_label = None
        self.last_accepted_label = None
        self.last_adaptive_corrected = False


# ----------------------------------
# QUICK VERIFICATION RUNNER
# ----------------------------------

if __name__ == "__main__":
    interpreter = SignInterpreter()

    print("Testing Adaptive Sign Correction (V -> DEL -> V -> DEL -> V)...")
    res1 = interpreter.update("V")
    print("Sign V:", res1)
    interpreter.update("NOTHING")

    res2 = interpreter.update("DEL")
    print("Sign DEL:", res2)
    interpreter.update("NOTHING")

    res3 = interpreter.update("V")
    print("Sign V (retry):", res3)
    interpreter.update("NOTHING")

    res4 = interpreter.update("DEL")
    print("Sign DEL (reset):", res4)
    interpreter.update("NOTHING")

    res5 = interpreter.update("V")
    print("Sign V (normal):", res5)

    print("\nFinal text:", interpreter.text)
