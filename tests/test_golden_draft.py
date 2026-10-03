from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_index.core import Document

from eval.make_golden_draft import draft_question, make_draft


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


class GoldenDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node = Document(
            text="Revenue increased by 12 percent during the quarter.",
            metadata={"Ticker": "TEST", "Company_Name": "Test Co.", "Quarter": "2024-Q1"},
        )

    def test_retries_when_gold_snippet_is_not_verbatim(self) -> None:
        llm = FakeLLM(
            [
                json.dumps({
                    "question": "What happened to revenue?",
                    "gold_snippet": "revenue rose 12%",
                    "type": "exact",
                }),
                json.dumps({
                    "question": "How did revenue change?",
                    "gold_snippet": "12 percent",
                    "type": "exact",
                }),
            ]
        )

        draft = draft_question(self.node, llm)

        self.assertEqual(draft["gold_snippet"], "12 percent")
        self.assertTrue(draft["needs_review"])
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("previous JSON response failed validation", llm.prompts[1])

    def test_fails_clearly_after_invalid_responses(self) -> None:
        llm = FakeLLM([
            '{"question":"Question one?","gold_snippet":"not present","type":"exact"}',
            '{"question":"Question two?","gold_snippet":"still absent","type":"exact"}',
        ])

        with self.assertRaisesRegex(ValueError, "after 2 attempts"):
            draft_question(self.node, llm, max_attempts=2)

    def test_valid_drafts_are_saved_and_later_samples_continue_after_failure(self) -> None:
        llm = FakeLLM([
            '{"question":"How did revenue change?","gold_snippet":"12 percent","type":"exact"}',
            '{"question":"Bad draft?","gold_snippet":"not present","type":"exact"}',
            '{"question":"Still bad?","gold_snippet":"absent","type":"exact"}',
            '{"question":"No match?","gold_snippet":"missing","type":"exact"}',
            '{"question":"How did revenue change?","gold_snippet":"12 percent","type":"exact"}',
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "golden.jsonl"
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
                patch(
                    "eval.make_golden_draft.load_data",
                    return_value=[self.node, self.node, self.node],
                ),
                patch("eval.make_golden_draft.OpenAI", return_value=llm),
            ):
                drafts = make_draft(
                    Path("unused.csv"),
                    output_path,
                    count=3,
                    seed=1,
                    model="test-model",
                )

            saved = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0]["gold_snippet"], "12 percent")
            self.assertEqual(saved[1]["gold_snippet"], "12 percent")
            self.assertEqual(len(drafts), 2)

    def test_resumes_after_existing_rows_without_regenerating_them(self) -> None:
        llm = FakeLLM([
            '{"question":"How did revenue change?","gold_snippet":"12 percent","type":"exact"}',
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "golden.jsonl"
            existing = {
                "question": "What was the revenue change?",
                "ticker": "TEST",
                "quarter": "2024-Q1",
                "gold_snippet": "12 percent",
                "type": "exact",
                "needs_review": True,
            }
            output_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
                patch("eval.make_golden_draft.load_data", return_value=[self.node, self.node]),
                patch("eval.make_golden_draft.OpenAI", return_value=llm),
            ):
                drafts = make_draft(
                    Path("unused.csv"),
                    output_path,
                    count=2,
                    seed=1,
                    model="test-model",
                )

            saved = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved), 2)
            self.assertEqual(len(drafts), 2)
            self.assertEqual(len(llm.prompts), 1)


if __name__ == "__main__":
    unittest.main()
