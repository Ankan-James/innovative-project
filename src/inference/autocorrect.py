
from spellchecker import SpellChecker


class AutoCorrect:
    """
    Lightweight spelling correction for SLMCI.

    Does not train an AI model.
    Preserves original words unless a correction
    is explicitly accepted.
    """

    def __init__(self, protected_words=None):
        self.spell = SpellChecker(
            language="en",
            distance=2
        )

        self.protected_words = {
            word.upper()
            for word in (protected_words or [])
        }

    def add_protected_word(self, word):
        """Protect a name or custom word."""
        self.protected_words.add(word.upper())

    def check_word(self, word):
        """
        Check a completed word and return
        a suggested correction if available.
        """

        original = word.strip()
        upper = original.upper()

        result = {
            "original": original,
            "suggestion": None,
            "status": "unchanged"
        }

        if not original:
            return result

        # Never change protected names or terms.
        if upper in self.protected_words:
            result["status"] = "protected"
            return result

        # Ignore numbers, punctuation and
        # very short words.
        if not original.isalpha() or len(original) < 3:
            return result

        lowercase = original.lower()

        # Already a recognized dictionary word.
        if not self.spell.unknown([lowercase]):
            result["status"] = "correct"
            return result

        suggestion = self.spell.correction(lowercase)

        if suggestion and suggestion != lowercase:
            result["suggestion"] = suggestion.upper()
            result["status"] = "review"
        else:
            result["status"] = "unknown"

        return result


# ========================================
# TEST WITHOUT WEBCAM
# ========================================

if __name__ == "__main__":

    autocorrect = AutoCorrect(
        protected_words={"ANKAN"}
    )

    test_words = [
        "HELKO",
        "WROLD",
        "ANKAN",
        "HELLO",
        "BIGGER"
    ]

    print("\nSLMCI Autocorrect Test\n")

    for word in test_words:
        result = autocorrect.check_word(word)

        print("Word:", result["original"])
        print("Suggestion:", result["suggestion"])
        print("Status:", result["status"])
        print("------------------------")

