"""
Unit tests for SLMCI Adaptive Sign Correction System
Tests all required scenarios without requiring a webcam.
"""

import unittest
from src.inference.sign_interpreter import SignInterpreter


class TestAdaptiveSignCorrection(unittest.TestCase):
    def setUp(self):
        self.interpreter = SignInterpreter()

    def _release_hand(self):
        """Simulate deliberate hand release / absence between sign attempts."""
        self.interpreter.update("NOTHING")

    def test_v_del_v_substitutes_k(self):
        """V -> DEL -> V -> K"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("V")

        self.assertEqual(self.interpreter.text, "K")
        self.assertEqual(res["label"], "K")
        self.assertEqual(res["raw_label"], "V")
        self.assertEqual(res["accepted_label"], "K")
        self.assertTrue(res["adaptive_corrected"])

    def test_k_del_k_substitutes_v(self):
        """K -> DEL -> K -> V"""
        self.interpreter.update("K")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("K")

        self.assertEqual(self.interpreter.text, "V")
        self.assertEqual(res["label"], "V")
        self.assertEqual(res["raw_label"], "K")
        self.assertEqual(res["accepted_label"], "V")
        self.assertTrue(res["adaptive_corrected"])

    def test_a_del_a_substitutes_t(self):
        """A -> DEL -> A -> T"""
        self.interpreter.update("A")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("A")

        self.assertEqual(self.interpreter.text, "T")
        self.assertEqual(res["label"], "T")
        self.assertEqual(res["raw_label"], "A")
        self.assertEqual(res["accepted_label"], "T")
        self.assertTrue(res["adaptive_corrected"])

    def test_t_del_t_substitutes_a(self):
        """T -> DEL -> T -> A"""
        self.interpreter.update("T")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("T")

        self.assertEqual(self.interpreter.text, "A")
        self.assertEqual(res["label"], "A")
        self.assertEqual(res["raw_label"], "T")
        self.assertEqual(res["accepted_label"], "A")
        self.assertTrue(res["adaptive_corrected"])

    def test_v_del_k_keeps_k(self):
        """V -> DEL -> K -> K (model prediction already equals intended alternative)"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("K")

        self.assertEqual(self.interpreter.text, "K")
        self.assertEqual(res["label"], "K")
        self.assertEqual(res["raw_label"], "K")
        self.assertEqual(res["accepted_label"], "K")
        self.assertFalse(res["adaptive_corrected"])

    def test_a_del_t_keeps_t(self):
        """A -> DEL -> T -> T (model prediction already equals intended alternative)"""
        self.interpreter.update("A")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("T")

        self.assertEqual(self.interpreter.text, "T")
        self.assertEqual(res["label"], "T")
        self.assertEqual(res["raw_label"], "T")
        self.assertEqual(res["accepted_label"], "T")
        self.assertFalse(res["adaptive_corrected"])

    def test_v_del_v_del_v_reverts_to_raw_v(self):
        """V -> DEL -> V -> DEL -> V -> Final V (reversible, no infinite loop)"""
        # 1. Sign V
        self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "V")
        self._release_hand()

        # 2. DEL
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # 3. Retry V -> substituted with K
        res_retry = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "K")
        self.assertEqual(res_retry["label"], "K")
        self.assertTrue(res_retry["adaptive_corrected"])
        self._release_hand()

        # 4. DEL again -> removes K and resets correction
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # 5. Next attempt V -> accepts raw AI prediction V
        res_final = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "V")
        self.assertEqual(res_final["label"], "V")
        self.assertEqual(res_final["raw_label"], "V")
        self.assertFalse(res_final["adaptive_corrected"])

    def test_a_del_a_del_a_reverts_to_raw_a(self):
        """A -> DEL -> A -> DEL -> A -> Final A (reversible, no infinite loop)"""
        # 1. Sign A
        self.interpreter.update("A")
        self.assertEqual(self.interpreter.text, "A")
        self._release_hand()

        # 2. DEL
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # 3. Retry A -> substituted with T
        res_retry = self.interpreter.update("A")
        self.assertEqual(self.interpreter.text, "T")
        self.assertEqual(res_retry["label"], "T")
        self.assertTrue(res_retry["adaptive_corrected"])
        self._release_hand()

        # 4. DEL again -> removes T and resets correction
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # 5. Next attempt A -> accepts raw AI prediction A
        res_final = self.interpreter.update("A")
        self.assertEqual(self.interpreter.text, "A")
        self.assertEqual(res_final["label"], "A")
        self.assertEqual(res_final["raw_label"], "A")
        self.assertFalse(res_final["adaptive_corrected"])

    def test_v_del_b_unrelated_cancels_correction(self):
        """V -> DEL -> B -> B (unrelated letter cancels pending correction)"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("B")

        self.assertEqual(self.interpreter.text, "B")
        self.assertEqual(res["label"], "B")
        self.assertEqual(res["raw_label"], "B")
        self.assertFalse(res["adaptive_corrected"])

    def test_b_del_b_non_confusion_pair(self):
        """B -> DEL -> B -> B (letters outside confusion pairs behave normally)"""
        self.interpreter.update("B")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("B")

        self.assertEqual(self.interpreter.text, "B")
        self.assertEqual(res["label"], "B")
        self.assertFalse(res["adaptive_corrected"])

    def test_del_on_empty_text(self):
        """DEL on empty text does not activate correction"""
        res_del = self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        res_v = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "V")
        self.assertEqual(res_v["label"], "V")
        self.assertFalse(res_v["adaptive_corrected"])

    def test_v_del_space_cancels_correction(self):
        """V -> DEL -> SPACE -> cancels correction"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        self.interpreter.update("SPACE")
        self._release_hand()

        res_v = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "V")
        self.assertEqual(res_v["label"], "V")
        self.assertFalse(res_v["adaptive_corrected"])

    def test_v_del_clear_resets_state(self):
        """V -> DEL -> clear() -> resets all state"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self.interpreter.clear()

        res_v = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "V")
        self.assertEqual(res_v["label"], "V")
        self.assertFalse(res_v["adaptive_corrected"])

    def test_held_del_over_multiple_updates(self):
        """Held DEL gesture triggers only one deletion"""
        self.interpreter.update("A")
        self._release_hand()
        self.interpreter.update("B")
        self.assertEqual(self.interpreter.text, "AB")

        # First DEL deletes B
        res1 = self.interpreter.update("DEL")
        self.assertTrue(res1["accepted"])
        self.assertEqual(self.interpreter.text, "A")

        # Held DEL updates do not delete further
        res2 = self.interpreter.update("DEL")
        self.assertFalse(res2["accepted"])
        self.assertEqual(self.interpreter.text, "A")

        res3 = self.interpreter.update("DEL")
        self.assertFalse(res3["accepted"])
        self.assertEqual(self.interpreter.text, "A")

    def test_held_v_over_multiple_updates(self):
        """Held V gesture accepts only one character"""
        res1 = self.interpreter.update("V")
        self.assertTrue(res1["accepted"])
        self.assertEqual(self.interpreter.text, "V")

        res2 = self.interpreter.update("V")
        self.assertFalse(res2["accepted"])
        self.assertEqual(self.interpreter.text, "V")

        res3 = self.interpreter.update("V")
        self.assertFalse(res3["accepted"])
        self.assertEqual(self.interpreter.text, "V")

    def test_held_corrected_v_preserves_display_label(self):
        """Held sign after adaptive correction preserves display label without repeating"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()

        # Retry V -> substituted with K
        res1 = self.interpreter.update("V")
        self.assertTrue(res1["accepted"])
        self.assertEqual(res1["label"], "K")
        self.assertEqual(self.interpreter.text, "K")

        # Continuing to hold V: repeat suppression active, display label stays K
        res2 = self.interpreter.update("V")
        self.assertFalse(res2["accepted"])
        self.assertEqual(res2["label"], "K")
        self.assertEqual(self.interpreter.text, "K")

    def test_ordinary_word_formation(self):
        """Ordinary word formation with spaces and deletions works properly"""
        for char in "HELLO":
            self.interpreter.update(char)
            self._release_hand()

        self.assertEqual(self.interpreter.text, "HELLO")

        self.interpreter.update("SPACE")
        self._release_hand()

        for char in "WORLD":
            self.interpreter.update(char)
            self._release_hand()

        self.assertEqual(self.interpreter.text, "HELLO WORLD")

    def test_word_with_correction(self):
        """Forming word 'THANK' when user signs 'THANV', deletes V, and signs V again"""
        for char in "THAN":
            self.interpreter.update(char)
            self._release_hand()

        # Accidental V prediction
        self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "THANV")
        self._release_hand()

        # DEL removes V
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "THAN")
        self._release_hand()

        # User retries sign, model produces V again -> substituted to K!
        res = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "THANK")
        self.assertEqual(res["label"], "K")
        self.assertTrue(res["adaptive_corrected"])


if __name__ == "__main__":
    unittest.main()
