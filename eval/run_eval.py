from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from alpha_detective import (
    DEFAULT_PERSIST_DIR,
    RetrievalConfig,
    build_hybrid_retriever,
    build_vector_index,
    load_data,
    retrieve_nodes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")
RESULTS_PATH = Path(__file__).with_name("results.csv")
QUESTION_TYPES = {"entity", "semantic", "exact", "aggregate"}

CONFIGS = (
    (
        "dense-only k=5",
        RetrievalConfig(mode="dense", retrieve_k=5, fused_k=5, final_k=5),
    ),
    (
        "bm25-only k=5",
        RetrievalConfig(mode="bm25", retrieve_k=5, fused_k=5, final_k=5),
    ),
    (
        "hybrid k=5 baseline",
        RetrievalConfig(mode="hybrid", retrieve_k=5, fused_k=5, final_k=5),
    ),
    (
        "hybrid wide (20->20->6)",
        RetrievalConfig(mode="hybrid", retrieve_k=20, fused_k=20, final_k=6),
    ),
    (
        "hybrid + filter",
        RetrievalConfig(
            mode="hybrid",
            retrieve_k=20,
            fused_k=20,
            final_k=6,
            use_filter=True,
        ),
    ),
    (
        "hybrid + filter + rerank (bge)",
        RetrievalConfig(
            mode="hybrid",
            retrieve_k=20,
            fused_k=20,
            final_k=6,
            use_filter=True,
            use_rerank=True,
            rerank_backend="bge",
        ),
    ),
    (
        "hybrid + filter + rerank (flashrank)",
        RetrievalConfig(
            mode="hybrid",
            retrieve_k=20,
            fused_k=20,
            final_k=6,
            use_filter=True,
            use_rerank=True,
            rerank_backend="flashrank",
        ),
    ),
)


def _load_golden(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}.")
            required = {
                "question",
                "ticker",
                "quarter",
                "gold_snippet",
                "type",
                "needs_review",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"Missing fields in {path} line {line_number}: {sorted(missing)}"
                )
            if row["type"] not in QUESTION_TYPES:
                raise ValueError(
                    f"Invalid question type in {path} line {line_number}: {row['type']!r}"
                )
            if not isinstance(row["needs_review"], bool):
                raise ValueError(
                    f"needs_review must be a boolean in {path} line {line_number}."
                )
            if not all(
                isinstance(row[field], str) and row[field].strip()
                for field in ("question", "ticker", "quarter", "gold_snippet")
            ):
                raise ValueError(
                    f"Question, ticker, quarter, and gold_snippet must be non-empty "
                    f"strings in {path} line {line_number}."
                )
            rows.append(row)
    return [row for row in rows if not row["needs_review"]]


def _summarize(
    config_name: str,
    question_type: str,
    outcomes: list[tuple[int | None, float]],
) -> dict[str, Any]:
    count = len(outcomes)
    reciprocal_ranks = [
        1 / rank if rank is not None else 0.0
        for rank, _ in outcomes
    ]
    return {
        "config": config_name,
        "question_type": question_type,
        "questions": count,
        "hit@1": sum(rank is not None and rank <= 1 for rank, _ in outcomes) / count,
        "hit@3": sum(rank is not None and rank <= 3 for rank, _ in outcomes) / count,
        "hit@5": sum(rank is not None and rank <= 5 for rank, _ in outcomes) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "p50_latency_ms": statistics.median(
            latency_seconds for _, latency_seconds in outcomes
        ) * 1000,
    }


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "question_type",
        "questions",
        "hit@1",
        "hit@3",
        "hit@5",
        "mrr",
        "p50_latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("config", "Config"),
        ("question_type", "Type"),
        ("questions", "N"),
        ("hit@1", "Hit@1"),
        ("hit@3", "Hit@3"),
        ("hit@5", "Hit@5"),
        ("mrr", "MRR"),
        ("p50_latency_ms", "p50 ms"),
    ]
    print("| " + " | ".join(label for _, label in columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if key in {"hit@1", "hit@3", "hit@5", "mrr"}:
                values.append(f"{value:.4f}")
            elif key == "p50_latency_ms":
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def run_eval(
    data_path: Path,
    golden_path: Path = GOLDEN_PATH,
    results_path: Path = RESULTS_PATH,
    persist_dir: Path = PROJECT_ROOT / DEFAULT_PERSIST_DIR,
) -> list[dict[str, Any]]:
    rows = _load_golden(golden_path)
    if not rows:
        raise ValueError(
            f"No reviewed questions found in {golden_path}; "
            "remove or set needs_review=false for reviewed rows."
        )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("Set OPENAI_API_KEY in the environment before running evaluation.")

    nodes = load_data(data_path)
    vector_index = build_vector_index(nodes, api_key, persist_dir=str(persist_dir))
    result_rows: list[dict[str, Any]] = []

    for config_name, config in CONFIGS:
        display_config_name = config_name
        retriever = build_hybrid_retriever(
            nodes,
            api_key,
            config=config,
            persist_dir=str(persist_dir),
            vector_index=vector_index,
        )
        outcomes: list[tuple[int | None, float]] = []
        outcomes_by_type: dict[str, list[tuple[int | None, float]]] = defaultdict(list)
        for row in rows:
            started = time.perf_counter()
            retrieved = retrieve_nodes(row["question"], retriever, config)
            elapsed = time.perf_counter() - started
            snippet = row["gold_snippet"].casefold()
            rank = next(
                (
                    position
                    for position, item in enumerate(retrieved, 1)
                    if snippet in item.node.get_content().casefold()
                ),
                None,
            )
            outcome = (rank, elapsed)
            outcomes.append(outcome)
            outcomes_by_type[row["type"]].append(outcome)

        if retriever.rerank_warning:
            display_config_name += " [rerank unavailable; fallback used]"
            print(f"WARNING: {config_name}: {retriever.rerank_warning}")

        result_rows.append(_summarize(display_config_name, "all", outcomes))
        for question_type, typed_outcomes in sorted(outcomes_by_type.items()):
            result_rows.append(
                _summarize(display_config_name, question_type, typed_outcomes)
            )

    _write_results(results_path, result_rows)
    _print_markdown(result_rows)
    return result_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against reviewed golden questions.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "earnings_transcripts.csv",
        help="CSV transcript dataset (default: earnings_transcripts.csv).",
    )
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PERSIST_DIR,
    )
    args = parser.parse_args()
    run_eval(
        data_path=args.data,
        golden_path=args.golden,
        results_path=args.results,
        persist_dir=args.persist_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
