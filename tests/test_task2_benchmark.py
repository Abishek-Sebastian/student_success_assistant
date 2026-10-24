"""Focused tests for the Task 2 four-condition benchmark."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from task2.src.evaluator import evaluate_one


class EvaluatorMemoryModeTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "explanation_id": "E_TEST",
            "lesson_id": "L1",
            "explanation_text": "Plants use sunlight to make food through photosynthesis.",
        }
        self.knowledge = {"L1": {"title": "Photosynthesis", "summary": "Plants use light to make food.", "key_points": ["photosynthesis"], "common_misconceptions": []}}

    @patch("task2.src.evaluator.call_chat_api", return_value='{"verdict":"understood","got_right":"correct","missed":"one detail","hint":"what is the role of light?"}')
    @patch("task2.src.evaluator.retrieve_cases")
    def test_no_memory_mode_never_calls_retrieval(self, retrieve_mock, _api_mock):
        result = evaluate_one(self.row, self.knowledge, max_retries=0, use_memory=False)
        retrieve_mock.assert_not_called()
        self.assertEqual(result["retrieved_cases"], 0)
        self.assertEqual(result["verdict"], "understood")

    @patch("task2.src.evaluator.call_chat_api", return_value='{"verdict":"understood","got_right":"correct","missed":"one detail","hint":"what is the role of light?"}')
    @patch("task2.src.evaluator.retrieve_cases", return_value=[{"explanation": "similar", "verdict": "understood"}])
    def test_memory_mode_calls_retrieval(self, retrieve_mock, _api_mock):
        result = evaluate_one(self.row, self.knowledge, max_retries=0, use_memory=True)
        retrieve_mock.assert_called_once()
        self.assertEqual(result["retrieved_cases"], 1)


if __name__ == "__main__":
    unittest.main()
