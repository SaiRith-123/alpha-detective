from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_index.core import Document

from eval.auto_review_golden import auto_review


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


class AutoReviewGoldenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = [
            Document(
                text="Quarterly revenue climbed to 12 percent after a strong launch.",
                metadata={"Ticker": "TEST", "Quarter": "2024-Q1"},
            ),
            Document(
                text="The company also expanded its hiring plan.",
                metadata={"Ticker": "TEST", "Quarter": "2024-Q1"},
            ),
        ]
        self.row = {
            "question": "What happened to quarterly revenue?",
            "ticker": "TEST",
            "quarter": "2024-Q1",
            "gold_snippet": "12 percent",
            "type": "exact",
            "needs_review": True,
        }

    def test_unique_exact_evidence_is_machine_checked_ready(self) -> None:
        llm = FakeLLM([
            json.dumps({
                "question": "How much did revenue rise?",
                "gold_snippet": "12 percent",
                "type": "exact",
                "source_index": 0,
            })
        ])
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            with patch("eval.auto_review_golden.load_data", return_value=self.nodes):
                approved, unresolved = auto_review(
                    input_path=input_path,
                    output_path=output_path,
                    data_path=Path("unused.csv"),
                    llm=llm,
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual((approved, unresolved), (1, 0))
            self.assertFalse(result["needs_review"])
            self.assertEqual(
                result["review_method"],
                "automated_llm_and_chunk_validation",
            )

    def test_duplicate_evidence_remains_unresolved(self) -> None:
        duplicate_nodes = [
            self.nodes[0],
            Document(
                text="Revenue reached 12 percent.",
                metadata={"Ticker": "OTHER", "Quarter": "2024-Q1"},
            ),
        ]
        llm = FakeLLM([
            json.dumps({
                "question": "How much did revenue rise?",
                "gold_snippet": "12 percent",
                "type": "exact",
                "source_index": 0,
            }),
        ] * 3)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            with patch("eval.auto_review_golden.load_data", return_value=duplicate_nodes):
                approved, unresolved = auto_review(
                    input_path=input_path,
                    output_path=output_path,
                    data_path=Path("unused.csv"),
                    attempts=3,
                    llm=llm,
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual((approved, unresolved), (0, 1))
            self.assertTrue(result["needs_review"])
            self.assertIn("occurs in 2 chunks", result["review_error"])


if __name__ == "__main__":
    unittest.main()
