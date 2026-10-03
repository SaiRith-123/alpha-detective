from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from alpha_detective import (
    QueryAwareRetriever,
    RerankerDependencyMissing,
    RetrievalConfig,
    _rerank_candidates,
)


def make_candidates() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=Document(text=f"Evidence {index}", metadata={"Ticker": "TEST"}),
            score=1.0 / index,
        )
        for index in range(1, 4)
    ]


class FakeBGEReranker:
    def postprocess_nodes(self, nodes, query_str):
        if query_str != "query":
            raise AssertionError("Unexpected reranker query.")
        for node in nodes:
            node.score = float(node.node.get_content().split()[-1])
        return list(reversed(nodes))

class FakeRetriever(BaseRetriever):
    def __init__(self, candidates: list[NodeWithScore]) -> None:
        super().__init__()
        self.candidates = candidates

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if not query_bundle.query_str:
            return []
        return self.candidates


class FakeFlashRank:
    def rerank(self, request):
        if request.query != "query" or len(request.passages) != 3:
            raise AssertionError("Unexpected FlashRank request.")
        return [
            {"id": 2, "score": 0.9},
            {"id": 1, "score": 0.7},
            {"id": 0, "score": 0.2},
        ]


class FakeRerankRequest:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages


class RerankingTests(unittest.TestCase):
    def test_rerank_preserves_rrf_and_stores_rerank_score(self) -> None:
        candidates = make_candidates()
        original_scores = {
            item.node.node_id: item.score
            for item in candidates
        }
        with patch(
            "alpha_detective._load_reranker",
            return_value=FakeBGEReranker(),
        ):
            reranked = _rerank_candidates("query", candidates, "bge")

        self.assertEqual(
            [item.node.get_content() for item in reranked],
            ["Evidence 3", "Evidence 2", "Evidence 1"],
        )
        for item in reranked:
            self.assertEqual(
                item.node.metadata["rrf_score"],
                original_scores[item.node.node_id],
            )
            self.assertEqual(
                item.node.metadata["rerank_score"],
                item.score,
            )

    def test_flashrank_uses_candidate_ids_and_preserves_rrf_scores(self) -> None:
        candidates = make_candidates()
        original_scores = {
            item.node.node_id: item.score
            for item in candidates
        }
        with patch(
            "alpha_detective._load_reranker",
            return_value=(FakeFlashRank(), FakeRerankRequest),
        ):
            reranked = _rerank_candidates("query", candidates, "flashrank")

        self.assertEqual(
            [item.node.get_content() for item in reranked],
            ["Evidence 3", "Evidence 2", "Evidence 1"],
        )
        for item in reranked:
            self.assertEqual(
                item.node.metadata["rrf_score"],
                original_scores[item.node.node_id],
            )
            self.assertEqual(
                item.node.metadata["rerank_score"],
                item.score,
            )

    def test_reranker_is_cached_per_backend(self) -> None:
        from alpha_detective import _RERANKER_CACHE, _load_reranker

        _RERANKER_CACHE.pop("flashrank", None)
        ranker = FakeFlashRank()
        ranker_init_calls = []

        def build_ranker(**kwargs):
            ranker_init_calls.append(kwargs)
            return ranker

        integration = SimpleNamespace(
            Ranker=build_ranker,
            RerankRequest=FakeRerankRequest,
        )
        with patch(
            "alpha_detective.importlib.import_module",
            return_value=integration,
        ) as import_module:
            first = _load_reranker("flashrank")
            second = _load_reranker("flashrank")

        self.assertIs(first, second)
        import_module.assert_called_once_with("flashrank")
        self.assertEqual(ranker_init_calls[0]["model_name"], "ms-marco-TinyBERT-L-2-v2")
        _RERANKER_CACHE.pop("flashrank", None)

    def test_missing_backend_falls_back_and_exposes_warning(self) -> None:
        candidates = make_candidates()
        retriever = QueryAwareRetriever(
            nodes=[],
            index=None,
            global_retriever=FakeRetriever(candidates),
            api_key="test-key",
            config=RetrievalConfig(
                use_rerank=True,
                rerank_backend="flashrank",
                final_k=2,
            ),
        )
        with patch(
            "alpha_detective._load_reranker",
            side_effect=RerankerDependencyMissing("Install optional extras."),
        ):
            results = retriever.retrieve("query")

        self.assertEqual(results, candidates[:2])
        self.assertEqual(retriever.rerank_warning, "Install optional extras.")
        for item in results:
            self.assertEqual(item.node.metadata["rrf_score"], item.score)


if __name__ == "__main__":
    unittest.main()
