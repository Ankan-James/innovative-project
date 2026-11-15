"""
Unit tests for SLMCI Adaptive Sign Correction System (All 26 Letters)
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

    # -------------------------------------------------------------------------
    # Bidirectional K / V tests
    # -------------------------------------------------------------------------

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

    def test_v_del_k_keeps_k(self):
        """V -> DEL -> K -> K (model prediction already equals alternative)"""
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

    def test_v_del_v_del_v_reverts_to_raw_v(self):
        """V -> DEL -> V -> DEL -> V -> Final V (reversible, no infinite loop)"""
        self.interpreter.update("V")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res_retry = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "K")
        self.assertTrue(res_retry["adaptive_corrected"])
        self._release_hand()

        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        res_final = self.interpreter.update("V")
        self.assertEqual(self.interpreter.text, "V")
        self.assertEqual(res_final["label"], "V")
        self.assertEqual(res_final["raw_label"], "V")
        self.assertFalse(res_final["adaptive_corrected"])

    # -------------------------------------------------------------------------
    # Bidirectional A / T tests
    # -------------------------------------------------------------------------

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

    def test_a_del_t_keeps_t(self):
        """A -> DEL -> T -> T (model prediction already equals alternative)"""
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

    def test_a_del_a_del_a_reverts_to_raw_a(self):
        """A -> DEL -> A -> DEL -> A -> Final A (reversible, no infinite loop)"""
        self.interpreter.update("A")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res_retry = self.interpreter.update("A")
        self.assertEqual(self.interpreter.text, "T")
        self.assertTrue(res_retry["adaptive_corrected"])
        self._release_hand()

        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        res_final = self.interpreter.update("A")
        self.assertEqual(self.interpreter.text, "A")
        self.assertEqual(res_final["label"], "A")
        self.assertEqual(res_final["raw_label"], "A")
        self.assertFalse(res_final["adaptive_corrected"])

    # -------------------------------------------------------------------------
    # One-way I -> J tests
    # -------------------------------------------------------------------------

    def test_i_del_i_substitutes_j(self):
        """I -> DEL -> I -> J"""
        self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "I")
        self._release_hand()

        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        res = self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "J")
        self.assertEqual(res["label"], "J")
        self.assertEqual(res["raw_label"], "I")
        self.assertEqual(res["accepted_label"], "J")
        self.assertTrue(res["adaptive_corrected"])

    def test_i_del_i_del_i_reverts_to_raw_i(self):
        """I -> DEL -> I -> DEL -> I -> Final I (reversible, no infinite loop)"""
        self.interpreter.update("I")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()

        # Retry produces J
        res_retry = self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "J")
        self.assertEqual(res_retry["label"], "J")
        self.assertTrue(res_retry["adaptive_corrected"])
        self._release_hand()

        # 2nd DEL deletes J and resets correction
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # Next attempt produces raw I
        res_final = self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "I")
        self.assertEqual(res_final["label"], "I")
        self.assertEqual(res_final["raw_label"], "I")
        self.assertFalse(res_final["adaptive_corrected"])

    # -------------------------------------------------------------------------
    # One-way X -> Z tests
    # -------------------------------------------------------------------------

    def test_x_del_x_substitutes_z(self):
        """X -> DEL -> X -> Z"""
        self.interpreter.update("X")
        self.assertEqual(self.interpreter.text, "X")
        self._release_hand()

        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        res = self.interpreter.update("X")
        self.assertEqual(self.interpreter.text, "Z")
        self.assertEqual(res["label"], "Z")
        self.assertEqual(res["raw_label"], "X")
        self.assertEqual(res["accepted_label"], "Z")
        self.assertTrue(res["adaptive_corrected"])

    def test_x_del_x_del_x_reverts_to_raw_x(self):
        """X -> DEL -> X -> DEL -> X -> Final X (reversible, no infinite loop)"""
        self.interpreter.update("X")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()

        # Retry produces Z
        res_retry = self.interpreter.update("X")
        self.assertEqual(self.interpreter.text, "Z")
        self.assertEqual(res_retry["label"], "Z")
        self.assertTrue(res_retry["adaptive_corrected"])
        self._release_hand()

        # 2nd DEL deletes Z and resets correction
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # Next attempt produces raw X
        res_final = self.interpreter.update("X")
        self.assertEqual(self.interpreter.text, "X")
        self.assertEqual(res_final["label"], "X")
        self.assertEqual(res_final["raw_label"], "X")
        self.assertFalse(res_final["adaptive_corrected"])

    # -------------------------------------------------------------------------
    # Unrelated letters & non-confusion letters
    # -------------------------------------------------------------------------

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

    def test_i_del_b_unrelated_cancels_correction(self):
        """I -> DEL -> B -> B (unrelated letter cancels pending I->J correction)"""
        self.interpreter.update("I")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("B")

        self.assertEqual(self.interpreter.text, "B")
        self.assertEqual(res["label"], "B")
        self.assertFalse(res["adaptive_corrected"])

    def test_b_del_b_non_confusion_pair(self):
        """B -> DEL -> B -> B (letters outside correction map behave normally)"""
        self.interpreter.update("B")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        res = self.interpreter.update("B")

        self.assertEqual(self.interpreter.text, "B")
        self.assertEqual(res["label"], "B")
        self.assertFalse(res["adaptive_corrected"])

    # -------------------------------------------------------------------------
    # Special cases: Empty text, SPACE, clear(), inconclusive None
    # -------------------------------------------------------------------------

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

    def test_none_prediction_does_not_create_new_attempt(self):
        """None predictions (inconclusive frames) do not trigger new attempts"""
        self.interpreter.update("A")
        # Inconclusive prediction with hand present
        res_none = self.interpreter.update(None, hand_detected=True)
        self.assertFalse(res_none["accepted"])
        # Same held sign A
        res_held = self.interpreter.update("A")
        self.assertFalse(res_held["accepted"])
        self.assertEqual(self.interpreter.text, "A")

    # -------------------------------------------------------------------------
    # Repeat suppression: Held DEL & Held letters
    # -------------------------------------------------------------------------

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

    def test_held_corrected_j_preserves_display_label(self):
        """Held sign after I->J correction preserves display label J without repeating"""
        self.interpreter.update("I")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()

        # Retry I -> substituted with J
        res1 = self.interpreter.update("I")
        self.assertTrue(res1["accepted"])
        self.assertEqual(res1["label"], "J")
        self.assertEqual(self.interpreter.text, "J")

        # Continuing to hold I: repeat suppression active, display label stays J
        res2 = self.interpreter.update("I")
        self.assertFalse(res2["accepted"])
        self.assertEqual(res2["label"], "J")
        self.assertEqual(self.interpreter.text, "J")

    def test_held_corrected_z_preserves_display_label(self):
        """Held sign after X->Z correction preserves display label Z without repeating"""
        self.interpreter.update("X")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()

        # Retry X -> substituted with Z
        res1 = self.interpreter.update("X")
        self.assertTrue(res1["accepted"])
        self.assertEqual(res1["label"], "Z")
        self.assertEqual(self.interpreter.text, "Z")

        # Continuing to hold X: repeat suppression active, display label stays Z
        res2 = self.interpreter.update("X")
        self.assertFalse(res2["accepted"])
        self.assertEqual(res2["label"], "Z")
        self.assertEqual(self.interpreter.text, "Z")

    # -------------------------------------------------------------------------
    # Word formation with J and Z
    # -------------------------------------------------------------------------

    def test_forming_word_just(self):
        """Forming word 'JUST' using I -> DEL -> I for 'J'"""
        # Form 'J' via I -> DEL -> I
        self.interpreter.update("I")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "J")
        self._release_hand()

        # Add U, S, T
        for char in "UST":
            self.interpreter.update(char)
            self._release_hand()

        self.assertEqual(self.interpreter.text, "JUST")

    def test_forming_word_zero(self):
        """Forming word 'ZERO' using X -> DEL -> X for 'Z'"""
        # Form 'Z' via X -> DEL -> X
        self.interpreter.update("X")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        self.interpreter.update("X")
        self.assertEqual(self.interpreter.text, "Z")
        self._release_hand()

        # Add E, R, O
        for char in "ERO":
            self.interpreter.update(char)
            self._release_hand()

        self.assertEqual(self.interpreter.text, "ZERO")

    def test_deleting_from_word_with_j_does_not_trigger_j_retry(self):
        """Deleting J from a normal word does not initiate J->? retry (J not in map)"""
        # Form "J"
        self.interpreter.update("I")
        self._release_hand()
        self.interpreter.update("DEL")
        self._release_hand()
        self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "J")
        self._release_hand()

        # Add "A"
        self.interpreter.update("A")
        self.assertEqual(self.interpreter.text, "JA")
        self._release_hand()

        # DEL deletes A -> starts retry for A/T
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "J")
        self._release_hand()

        # DEL deletes J -> J is not in CORRECTION_MAP, so no retry for J
        self.interpreter.update("DEL")
        self.assertEqual(self.interpreter.text, "")
        self._release_hand()

        # Signing I produces I (normal)
        res = self.interpreter.update("I")
        self.assertEqual(self.interpreter.text, "I")
        self.assertEqual(res["label"], "I")
        self.assertFalse(res["adaptive_corrected"])


if __name__ == "__main__":
    unittest.main()
