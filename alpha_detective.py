"""
alpha_detective.py — Alpha-Detective Hybrid Search RAG Engine
"""

from __future__ import annotations
import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence, Tuple


from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import pandas as pd
from chromadb.errors import NotFoundError

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
TOP_K = 5
FUSION_NUM_QUERIES = 1
CHROMA_COLLECTION = "alpha_detective"
DEFAULT_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64

STRICT_PROMPT = "Given the context information and not prior knowledge, answer the query. If the context doesn't contain the answer, say 'Information not found in transcripts.'\nContext:\n{context}\n\nQuery: {query}\nAnswer:"

@dataclass(frozen=True)
class RetrievalConfig:
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    retrieve_k: int = 20
    fused_k: int = 20
    final_k: int = 6
    use_filter: bool = False
    use_rerank: bool = False
    rerank_backend: Literal["bge", "flashrank"] = "bge"

    def __post_init__(self) -> None:
        if self.mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}.")
        if self.rerank_backend not in {"bge", "flashrank"}:
            raise ValueError(f"Unsupported reranker backend: {self.rerank_backend}.")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (self.retrieve_k, self.fused_k, self.final_k)
        ):
            raise ValueError("Retrieval cutoffs must be positive integers.")


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()
_RERANKER_CACHE: dict[str, Any] = {}
_RERANKER_UNAVAILABLE = object()
_RERANKER_ERRORS: dict[str, str] = {}

class RerankerDependencyMissing(ImportError):
    pass

def _load_reranker(backend: str) -> Any:
    cached = _RERANKER_CACHE.get(backend)
    if cached is _RERANKER_UNAVAILABLE:
        raise RerankerDependencyMissing(_RERANKER_ERRORS[backend])
    if cached is not None:
        return cached

    try:
        if backend == "bge":
            integration = importlib.import_module(
                "llama_index.postprocessor.sbert_rerank"
            )

            reranker = integration.SentenceTransformerRerank(
                model="BAAI/bge-reranker-base",
                top_n=1000,
            )
        elif backend == "flashrank":
            integration = importlib.import_module("flashrank")

            reranker = (
                integration.Ranker(
                    model_name="ms-marco-TinyBERT-L-2-v2",
                    max_length=512,
                ),
                integration.RerankRequest,
            )
        else:
            raise ValueError(f"Unsupported reranker backend: {backend}.")
    except ImportError as exc:
        install_hint = (
            "Install requirements-rerank.txt to enable optional reranking."
        )
        message = f"The {backend} reranker dependencies are unavailable. {install_hint}"
        _RERANKER_CACHE[backend] = _RERANKER_UNAVAILABLE
        _RERANKER_ERRORS[backend] = message
        raise RerankerDependencyMissing(message) from exc

    _RERANKER_CACHE[backend] = reranker
    return reranker

def _rerank_candidates(
    query: str,
    candidates: Sequence[NodeWithScore],
    backend: str,
) -> list[NodeWithScore]:
    if not candidates:
        return []

    original_scores = {
        item.node.node_id: item.score
        for item in candidates
    }
    for item in candidates:
        if item.score is not None:
            item.node.metadata["rrf_score"] = item.score

    reranker = _load_reranker(backend)
    if backend == "bge":
        reranked = reranker.postprocess_nodes(list(candidates), query_str=query)
    else:
        ranker, request_class = reranker

        passages = [
            {
                "id": position,
                "text": item.node.get_content(),
                "meta": {},
            }
            for position, item in enumerate(candidates)
        ]
        results = ranker.rerank(request_class(query=query, passages=passages))
        reranked = [
            NodeWithScore(
                node=candidates[int(result["id"])].node,
                score=float(result["score"]),
            )
            for result in results
        ]

    for item in reranked:
        if item.score is not None:
            original_score = original_scores.get(item.node.node_id)
            if original_score is not None:
                item.node.metadata["rrf_score"] = original_score
            item.node.metadata["rerank_score"] = item.score
    return reranked

_QUARTER_PATTERN = re.compile(r"(?<!\w)((?:19|20)\d{2}-Q[1-4])(?!\w)", re.IGNORECASE)
_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b(?:inc(?:orporated)?|corp(?:oration)?|ltd|limited|plc|co(?:mpany)?)\.?\s*$",
    re.IGNORECASE,
)

def _normalize_company_name(name: str) -> str:
    normalized = re.sub(r"[^\w]+", " ", name.casefold(), flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    while True:
        without_suffix = _COMPANY_SUFFIX_PATTERN.sub("", normalized).strip()
        if without_suffix == normalized:
            return normalized
        normalized = without_suffix

class EntityIndex:
    def __init__(self, nodes: Sequence) -> None:
        self.company_tickers: dict[str, set[str]] = {}
        self.tickers: set[str] = set()
        for node in nodes:
            ticker = str(node.metadata.get("Ticker", "")).strip().upper()
            company = str(node.metadata.get("Company_Name", "")).strip()
            if ticker:
                self.tickers.add(ticker)
            normalized_company = _normalize_company_name(company)
            if normalized_company and ticker:
                self.company_tickers.setdefault(normalized_company, set()).add(ticker)

    def detect_entities(self, query: str) -> tuple[list[str], list[str]]:
        normalized_query = " ".join(
            re.sub(r"[^\w]+", " ", query.casefold(), flags=re.UNICODE).split()
        )
        tickers: set[str] = set()
        for company, company_tickers in self.company_tickers.items():
            pattern = rf"(?<!\w){re.escape(company)}(?!\w)"
            if re.search(pattern, normalized_query):
                tickers.update(company_tickers)

        for ticker in self.tickers:
            if re.search(rf"(?<!\w){re.escape(ticker)}(?!\w)", query):
                tickers.add(ticker)

        quarters = sorted(
            {match.group(1).upper() for match in _QUARTER_PATTERN.finditer(query)}
        )
        return sorted(tickers), quarters

def _ensure_api_key(api_key: str) -> None:
    if not api_key.strip() or api_key == "YOUR_OPENAI_API_KEY_HERE":
        raise RuntimeError("OPENAI_API_KEY is not set.")

def _nodes_fingerprint(nodes: list) -> str:
    content = [(node.get_content(), node.metadata) for node in nodes]
    serialized = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Ticker", "Company_Name", "Quarter", "Text"}
    normalized_columns: dict[str, list] = {}
    for column in df.columns:
        key = str(column).strip().casefold().replace(" ", "_")
        normalized_columns.setdefault(key, []).append(column)
    required_keys = {column.casefold() for column in required}
    if any(len(normalized_columns.get(key, [])) > 1 for key in required_keys):
        raise ValueError("CSV contains duplicate required column names.")
    rename_columns = {
        normalized_columns[column.casefold()][0]: column
        for column in required
        if normalized_columns.get(column.casefold())
    }
    return df.rename(columns=rename_columns)

def load_data(data) -> list:
    if isinstance(data, pd.DataFrame):
        df = _canonicalize_columns(data)
    else:
        sources = [data] if isinstance(data, (str, os.PathLike)) else list(data)
        frames = []
        for source in sources:
            if isinstance(source, (str, os.PathLike)):
                source_name = os.fspath(source)
                if not os.path.exists(source_name):
                    raise FileNotFoundError(f"'{source_name}' missing.")
                extension = os.path.splitext(source_name)[1].lower()
                if extension == ".txt":
                    with open(source_name, "r", encoding="utf-8", errors="ignore") as file:
                        text_content = file.read()
                    company_name = os.path.splitext(os.path.basename(source_name))[0]
                    frames.append(pd.DataFrame([{
                        "Ticker": company_name,
                        "Company_Name": company_name,
                        "Quarter": "N/A",
                        "Text": text_content,
                    }]))
                elif extension == ".csv":
                    frames.append(pd.read_csv(source_name))
                else:
                    raise ValueError(f"Unsupported transcript file type: '{source_name}'.")
            else:
                source_name = getattr(source, "name", "")
                extension = os.path.splitext(source_name)[1].lower()
                if extension == ".txt":
                    source.seek(0)
                    text_content = source.read().decode("utf-8", errors="ignore")
                    company_name = os.path.splitext(os.path.basename(source_name))[0]
                    frames.append(pd.DataFrame([{
                        "Ticker": company_name,
                        "Company_Name": company_name,
                        "Quarter": "N/A",
                        "Text": text_content,
                    }]))
                elif extension == ".csv":
                    source.seek(0)
                    frames.append(pd.read_csv(source))
                else:
                    raise ValueError(f"Unsupported transcript file type: '{source_name}'.")

        if not frames:
            raise ValueError("No transcript files provided.")
        df = pd.concat(
            [_canonicalize_columns(frame) for frame in frames],
            ignore_index=True,
        )

    required = {"Ticker", "Company_Name", "Quarter", "Text"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df = df[df["Text"].notna() & (df["Text"].str.strip() != "")].copy()
    if df.empty: raise ValueError("No non-empty transcript rows found.")

    parser = SimpleNodeParser.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes: list = []
    for _, row in df.iterrows():
        metadata = {"Ticker": str(row["Ticker"]), "Company_Name": str(row["Company_Name"]), "Quarter": str(row["Quarter"])}
        doc = Document(text=str(row["Text"]), metadata=metadata)
        nodes.extend(parser.get_nodes_from_documents([doc]))
    return nodes

def build_vector_index(
    nodes: list,
    api_key: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> VectorStoreIndex:
    _ensure_api_key(api_key)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=api_key)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    fingerprint = _nodes_fingerprint(nodes)
    collection_name = f"{CHROMA_COLLECTION}_{fingerprint[:40]}"
    collections = {collection.name: collection for collection in chroma_client.list_collections()}

    if collection_name in collections:
        chroma_collection = chroma_client.get_collection(name=collection_name)
        metadata = chroma_collection.metadata or {}
        if (
            metadata.get("index_fingerprint") != fingerprint
            or metadata.get("index_status") != "complete"
            or chroma_collection.count() != len(nodes)
        ):
            try:
                chroma_client.delete_collection(name=collection_name)
            except NotFoundError:
                pass
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"index_fingerprint": fingerprint, "index_status": "building"},
            )
            rebuild_index = True
        else:
            rebuild_index = False
    else:
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"index_fingerprint": fingerprint, "index_status": "building"},
        )
        rebuild_index = True

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if rebuild_index:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
        chroma_collection.modify(
            metadata={"index_fingerprint": fingerprint, "index_status": "complete"}
        )
    else:
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index

def _make_hybrid_retriever(
    dense_retriever: BaseRetriever,
    sparse_retrievers: Sequence[BaseRetriever],
    api_key: str,
    config: RetrievalConfig,
) -> QueryFusionRetriever:
    return QueryFusionRetriever(
        [dense_retriever, *sparse_retrievers],
        llm=OpenAI(model=LLM_MODEL, api_key=api_key),
        similarity_top_k=config.fused_k,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=FUSION_NUM_QUERIES,
    )

class QueryAwareRetriever:
    def __init__(
        self,
        nodes: Sequence,
        index: VectorStoreIndex | None,
        global_retriever: BaseRetriever,
        api_key: str,
        config: RetrievalConfig,
    ) -> None:
        self.nodes = nodes
        self.index = index
        self.global_retriever = global_retriever
        self.api_key = api_key
        self.config = config
        self.entity_index = EntityIndex(nodes)
        self._bm25_cache: dict[tuple[str | None, tuple[str, ...]], BaseRetriever] = {}
        self.rerank_warning: str | None = None

    def _finish_candidates(
        self,
        query: str,
        candidates: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        if not self.config.use_rerank:
            return candidates
        try:
            return _rerank_candidates(
                query,
                candidates[:self.config.fused_k],
                self.config.rerank_backend,
            )[:self.config.final_k]
        except RerankerDependencyMissing as exc:
            self.rerank_warning = str(exc)
            if self.config.mode == "hybrid":
                for item in candidates:
                    if item.score is not None:
                        item.node.metadata.setdefault("rrf_score", item.score)
            return candidates[:self.config.final_k]

    def _get_filtered_bm25(
        self,
        ticker: str | None,
        quarters: Sequence[str],
    ) -> BaseRetriever | None:
        key = (ticker, tuple(sorted(quarters)))
        if key not in self._bm25_cache:
            filtered_nodes = [
                node
                for node in self.nodes
                if (
                    ticker is None
                    or str(node.metadata.get("Ticker", "")).strip().upper() == ticker
                )
                and (
                    not quarters
                    or str(node.metadata.get("Quarter", "")).strip().upper() in quarters
                )
            ]
            if not filtered_nodes:
                return None
            self._bm25_cache[key] = BM25Retriever.from_defaults(
                nodes=filtered_nodes,
                similarity_top_k=self.config.retrieve_k,
            )
        return self._bm25_cache[key]

    @staticmethod
    def _retrieve_from_multiple(
        retrievers: Sequence[BaseRetriever],
        query: str,
    ) -> list[NodeWithScore]:
        merged: dict[str, NodeWithScore] = {}
        for retriever in retrievers:
            for item in retriever.retrieve(query):
                node_id = item.node.node_id
                previous = merged.get(node_id)
                if previous is None or (item.score or 0.0) > (previous.score or 0.0):
                    merged[node_id] = item
        return sorted(
            merged.values(),
            key=lambda item: item.score or 0.0,
            reverse=True,
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        if not self.config.use_filter:
            return self._finish_candidates(
                query,
                self.global_retriever.retrieve(query),
            )

        tickers, quarters = self.entity_index.detect_entities(query)
        if not tickers and not quarters:
            return self._finish_candidates(
                query,
                self.global_retriever.retrieve(query),
            )

        filters = []
        if tickers:
            filters.append(
                MetadataFilter(
                    key="Ticker",
                    value=tickers,
                    operator=FilterOperator.IN,
                )
            )
        if quarters:
            filters.append(
                MetadataFilter(
                    key="Quarter",
                    value=quarters,
                    operator=FilterOperator.IN,
                )
            )

        filtered_dense = (
            self.index.as_retriever(
                similarity_top_k=self.config.retrieve_k,
                filters=MetadataFilters(
                    filters=filters,
                    condition=FilterCondition.AND,
                ),
            )
            if self.index is not None and self.config.mode != "bm25"
            else None
        )
        sparse_retrievers = [
            retriever
            for ticker in (tickers or [None])
            if (retriever := self._get_filtered_bm25(ticker, quarters)) is not None
        ]

        if self.config.mode == "dense":
            candidates = (
                filtered_dense.retrieve(query) if filtered_dense is not None else []
            )
        elif self.config.mode == "bm25":
            candidates = self._retrieve_from_multiple(sparse_retrievers, query)
        elif filtered_dense is None:
            candidates = []
        else:
            filtered_fusion = _make_hybrid_retriever(
                filtered_dense,
                sparse_retrievers,
                self.api_key,
                self.config,
            )
            candidates = filtered_fusion.retrieve(query)
        return self._finish_candidates(query, candidates)

def build_hybrid_retriever(
    nodes: list,
    api_key: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
    vector_index: VectorStoreIndex | None = None,
) -> QueryAwareRetriever:
    _ensure_api_key(api_key)

    if config.mode == "bm25":
        sparse_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=config.retrieve_k,
        )
        global_retriever = sparse_retriever
    else:
        if vector_index is None:
            vector_index = build_vector_index(nodes, api_key, persist_dir)
        dense_retriever = vector_index.as_retriever(
            similarity_top_k=config.retrieve_k,
        )
        if config.mode == "dense":
            global_retriever = dense_retriever
        else:
            sparse_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=config.retrieve_k,
            )
            global_retriever = _make_hybrid_retriever(
                dense_retriever,
                [sparse_retriever],
                api_key,
                config,
            )

    return QueryAwareRetriever(
        nodes=nodes,
        index=vector_index,
        global_retriever=global_retriever,
        api_key=api_key,
        config=config,
    )

def retrieve_nodes(
    query: str,
    retriever: QueryAwareRetriever,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> list:
    retrieved_nodes = retriever.retrieve(query)[:config.final_k]
    if config.mode == "hybrid":
        for item in retrieved_nodes:
            if item.score is not None and not config.use_rerank:
                item.node.metadata["rrf_score"] = item.score
    return retrieved_nodes

def query_system(
    query: str,
    hybrid_retriever: QueryAwareRetriever,
    api_key: str,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> Tuple[str, list]:
    _ensure_api_key(api_key)
    retrieved_nodes = retrieve_nodes(query, hybrid_retriever, config)
    context = "\n\n".join(f"[Source: {n.node.metadata.get('Company_Name', '?')} ({n.node.metadata.get('Ticker', '?')}, {n.node.metadata.get('Quarter', '?')})]\n{n.node.get_content()}" for n in retrieved_nodes)
    prompt = STRICT_PROMPT.format(context=context, query=query)
    response = OpenAI(model=LLM_MODEL, api_key=api_key).complete(prompt)
    return str(response), retrieved_nodes