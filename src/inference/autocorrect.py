
import math
from collections import defaultdict
from spellchecker import SpellChecker


def edit_distance(a, b):
    """Count insertions, deletions, replacements and adjacent swaps."""
    rows = len(a) + 1
    cols = len(b) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = int(a[i - 1] != b[j - 1])

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                dp[i][j] = min(
                    dp[i][j],
                    dp[i - 2][j - 2] + 1
                )

    return dp[-1][-1]


class AutoCorrect:

    def __init__(self, protected_words=None):
        self.spell = SpellChecker(
            language="en",
            distance=2
        )

        self.protected_words = {
            word.casefold()
            for word in (protected_words or [])
        }

        # Index common anagrams so two swaps can
        # still produce useful correction candidates.
        self.anagrams = defaultdict(set)

        for word in self.spell.word_frequency.words():
            frequency = self.spell.word_frequency[word]

            if (
                4 <= len(word) <= 16
                and word.isalpha()
                and frequency >= 10000
            ):
                key = "".join(sorted(word))
                self.anagrams[key].add(word)

    def add_protected_word(self, word):
        self.protected_words.add(word.casefold())

    def check_word(self, word, name_mode=False):
        original = word.strip()
        lower = original.casefold()

        result = {
            "original": original,
            "corrected": original,
            "suggestion": None,
            "status": "unchanged"
        }

        if (
            not original.isalpha()
            or len(original) < 4
            or len(original) > 16
        ):
            return result

        if name_mode or lower in self.protected_words:
            result["status"] = "protected"
            return result

        if not self.spell.unknown([lower]):
            result["status"] = "correct"
            return result


        # Handle an accidental extra letter at the end.
        # Example: GINGERT -> GINGER
        if len(lower) >= 5:
            shorter_word = lower[:-1]

            if (
                not self.spell.unknown([shorter_word])
                and self.spell.word_frequency[shorter_word] >= 500
            ):
                if original.isupper():
                    corrected = shorter_word.upper()
                elif original.istitle():
                    corrected = shorter_word.capitalize()
                else:
                    corrected = shorter_word

                result["corrected"] = corrected
                result["suggestion"] = corrected
                result["status"] = "applied"

                return result

        
        candidates = set(
            self.spell.candidates(lower) or []
        )

        # Include anagrams for errors such as
        # WRLOD -> WORLD.
        key = "".join(sorted(lower))
        candidates.update(self.anagrams.get(key, set()))

        ranked = []

        for candidate in candidates:
            candidate = candidate.casefold()

            if candidate == lower:
                continue

            distance = edit_distance(lower, candidate)

            if distance < 1 or distance > 2:
                continue

            frequency = self.spell.word_frequency[candidate]

            minimum = (
                10000 if distance == 1 else 25000
            )

            if frequency < minimum:
                continue

            score = (
                math.log1p(frequency)
                - 2.2 * distance
                - 0.6 * abs(len(candidate) - len(lower))
            )

            if sorted(candidate) == sorted(lower):
                score += 0.5

            ranked.append((score, candidate))

        if not ranked:
            result["status"] = "unknown"
            return result

        ranked.sort(reverse=True)

        best_score, best_word = ranked[0]

        suggestion = (
            best_word.upper()
            if original.isupper()
            else best_word
        )

        result["suggestion"] = suggestion

        # Avoid automatically choosing between
        # similarly plausible corrections.
        if (
            len(ranked) > 1
            and best_score - ranked[1][0] < 0.5
        ):
            result["status"] = "ambiguous"
            return result

        result["corrected"] = suggestion
        result["status"] = "applied"

        return result


if __name__ == "__main__":
    autocorrect = AutoCorrect()

    for word in [
        "RELLO",
        "WROLD",
        "WRLOD",
        "BIGER",
        "ANKAN",
        "HELLO"
    ]:
        print(word, "->", autocorrect.check_word(word))

    print(
        "Name Mode:",
        autocorrect.check_word("ANKAN", name_mode=True)
    )
