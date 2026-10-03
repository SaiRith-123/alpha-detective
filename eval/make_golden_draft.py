from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from llama_index.llms.openai import OpenAI

from alpha_detective import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("golden.jsonl")
VALID_TYPES = {"entity", "semantic", "exact", "aggregate"}
MAX_DRAFT_ATTEMPTS = 3


class DraftLLM(Protocol):
    def complete(self, prompt: str) -> Any: ...


def sample_chunks(nodes: list, limit: int, seed: int) -> list:
    if limit < 1:
        raise ValueError("Sample count must be positive.")
    by_company: dict[str, list] = defaultdict(list)
    for node in nodes:
        company = str(node.metadata.get("Company_Name", ""))
        by_company[company].append(node)

    rng = random.Random(seed)
    for company_nodes in by_company.values():
        rng.shuffle(company_nodes)
    companies = sorted(by_company)
    rng.shuffle(companies)

    samples = []
    while len(samples) < min(limit, len(nodes)):
        added_this_round = False
        for company in companies:
            if by_company[company]:
                samples.append(by_company[company].pop())
                added_this_round = True
                if len(samples) == limit:
                    break
        if not added_this_round:
            break
    return samples


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _validate_draft(response: str, node: Any) -> dict[str, Any]:
    chunk = node.get_content()
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.IGNORECASE)
    draft = json.loads(response)
    question = draft.get("question")
    gold_snippet = draft.get("gold_snippet")
    question_type = draft.get("type")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("LLM draft has no non-empty question.")
    if not isinstance(gold_snippet, str) or not gold_snippet.strip():
        raise ValueError("LLM draft has no non-empty gold_snippet.")
    if gold_snippet.casefold() not in chunk.casefold():
        raise ValueError("LLM gold_snippet does not appear in its source chunk.")
    normalized_question = _normalize(question)
    if normalized_question and normalized_question in _normalize(chunk):
        raise ValueError("LLM question copied text from its source chunk.")
    if question_type not in VALID_TYPES:
        raise ValueError(f"LLM returned invalid question type: {question_type!r}.")
    return {
        "question": question.strip(),
        "ticker": str(node.metadata.get("Ticker", "")),
        "quarter": str(node.metadata.get("Quarter", "")),
        "gold_snippet": gold_snippet.strip(),
        "type": question_type,
        "needs_review": True,
    }

def draft_question(
    node: Any,
    llm: DraftLLM,
    max_attempts: int = MAX_DRAFT_ATTEMPTS,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    chunk = node.get_content()
    prompt = (
        "Create one factual evaluation question using only this transcript chunk. "
        "Paraphrase the information: do not copy a sentence or distinctive phrase "
        "from the chunk into the question. The gold_snippet must be a short, exact, "
        "contiguous substring copied verbatim from the chunk, including its exact "
        "spelling and punctuation. Return exactly one JSON object with keys "
        '"question", "gold_snippet", and "type"; type must be one of '
        '"entity", "semantic", "exact", or "aggregate". Do not infer facts '
        "that are not stated in the chunk.\n\n"
        f"Transcript chunk:\n{chunk}"
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        response = str(llm.complete(prompt))
        try:
            return _validate_draft(response, node)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                prompt = (
                    "Your previous JSON response failed validation: "
                    f"{exc}. Generate a corrected response. The gold_snippet must "
                    "be an exact, contiguous, verbatim substring of the source "
                    "chunk. The question must be paraphrased and not copied from "
                    "the chunk. Return only the required JSON object.\n\n"
                    f"Transcript chunk:\n{chunk}"
                )

    raise ValueError(
        f"Could not draft a valid question after {max_attempts} attempts: "
        f"{last_error}"
    ) from last_error


def make_draft(
    data_path: Path,
    output_path: Path,
    count: int,
    seed: int,
    model: str,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("Set OPENAI_API_KEY in the environment before drafting questions.")
    nodes = load_data(data_path)
    samples = sample_chunks(nodes, count, seed)
    llm = OpenAI(model=model, api_key=api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_drafts: list[dict[str, Any]] = []
    if output_path.exists() and not overwrite:
        with output_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    existing_drafts.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cannot resume: invalid JSON on line {line_number} "
                        f"of {output_path}. Use --overwrite to replace it."
                    ) from exc
        if len(existing_drafts) > len(samples):
            raise ValueError(
                f"Cannot resume: {output_path} contains {len(existing_drafts)} rows "
                f"but this run samples only {len(samples)} chunks. Use --overwrite "
                "to replace it."
            )
        for index, draft in enumerate(existing_drafts):
            node = samples[index]
            if (
                draft.get("ticker") != str(node.metadata.get("Ticker", ""))
                or draft.get("quarter") != str(node.metadata.get("Quarter", ""))
                or not isinstance(draft.get("gold_snippet"), str)
                or draft["gold_snippet"].casefold() not in node.get_content().casefold()
            ):
                raise ValueError(
                    f"Cannot resume: row {index + 1} of {output_path} does not match "
                    "the selected sample order. Use --overwrite to start over."
                )

    drafts = list(existing_drafts)
    open_mode = "w" if overwrite else "a"
    with output_path.open(open_mode, encoding="utf-8") as file:
        if existing_drafts and output_path.stat().st_size and not output_path.read_bytes().endswith(b"\n"):
            file.write("\n")
        for sample_number, node in enumerate(samples[len(existing_drafts):], len(existing_drafts) + 1):
            try:
                draft = draft_question(node, llm)
            except ValueError as exc:
                print(
                    f"Skipping sample {sample_number}/{len(samples)} after draft "
                    f"validation failed: {exc}",
                    file=sys.stderr,
                )
                continue
            file.write(json.dumps(draft, ensure_ascii=False) + "\n")
            file.flush()
            drafts.append(draft)
    return drafts


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft human-reviewable retrieval questions.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "earnings_transcripts.csv",
        help="CSV transcript dataset (default: earnings_transcripts.csv).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output instead of resuming it.",
    )
    args = parser.parse_args()
    drafts = make_draft(
        args.data, args.output, args.count, args.seed, args.model, args.overwrite
    )
    print(f"Saved {len(drafts)} review-required questions to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
