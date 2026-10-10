"""Unit tests for the strict boxed-answer scorer: python -m unittest -v test_scoring.py"""
import unittest

from check_pair import scorer


class ScoringTests(unittest.TestCase):
    def score(self, response, gold="315"):
        return scorer.process_results({"answer": "#### " + gold}, [response])["exact_match"]

    def test_final_answer_excludes_reasoning(self):
        self.assertEqual(self.score(r"<think>The total is 1800. \boxed{1800}</think>Final: \boxed{315}"), 1)

    def test_never_uses_intermediate_number(self):
        self.assertEqual(self.score("The total is 315. I could not finish."), 0)

    def test_rejects_undecoded_output(self):
        with self.assertRaises(ValueError):
            self.score(r"ĊFinalĠanswer: \boxed{315}")

    def test_incomplete_answers(self):
        for response in (r"\boxed{315", r"<think>\boxed{315}", r"\boxed{315} then \boxed{"):
            with self.subTest(response=response):
                self.assertEqual(self.score(response), 0)

    def test_full_numeric_match(self):
        self.assertEqual(self.score(r"\boxed{3,150.00}", "3150"), 1)
        self.assertEqual(self.score(r"\boxed{-12}", "-12"), 1)
        self.assertEqual(self.score(r"\boxed{1+314}"), 0)
        self.assertEqual(self.score(r"\boxed{315.00001}"), 0)

    def test_invalid_target(self):
        with self.assertRaises(ValueError):
            self.score(r"\boxed{315}", "unknown")


if __name__ == "__main__":
    unittest.main()
