"""
alpha_detective.py — Alpha-Detective Hybrid Search RAG Engine
"""

from __future__ import annotations
import os
from typing import Tuple


from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import pandas as pd

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
TOP_K = 5
FUSION_NUM_QUERIES = 1
CHROMA_COLLECTION = "alpha_detective"
DEFAULT_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64

STRICT_PROMPT = "Given the context information and not prior knowledge, answer the query. If the context doesn't contain the answer, say 'Information not found in transcripts.'\nContext:\n{context}\n\nQuery: {query}\nAnswer:"

def _ensure_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY", "") in ["", "YOUR_OPENAI_API_KEY_HERE"]:
        raise RuntimeError("OPENAI_API_KEY is not set.")

def load_data(data) -> list:
    if isinstance(data, str):
        if not os.path.exists(data): raise FileNotFoundError(f"'{data}' missing.")
        if data.endswith('.txt'):
            with open(data, 'r', encoding='utf-8', errors='ignore') as f: text_content = f.read()
            company_name = os.path.splitext(os.path.basename(data))[0]
            df = pd.DataFrame([{"Ticker": company_name, "Company_Name": company_name, "Quarter": "N/A", "Text": text_content}])
        else:
            df = pd.read_csv(data)
            if len(df.columns) >= 4: df.columns = ['Ticker', 'Company_Name', 'Quarter', 'Text'] + list(df.columns[4:])
    else:
        df = data

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

def build_hybrid_retriever(nodes: list, persist_dir: str = DEFAULT_PERSIST_DIR, reset_collection: bool = False) -> QueryFusionRetriever:
    _ensure_api_key()
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    if reset_collection:
        try: chroma_client.delete_collection(name=CHROMA_COLLECTION)
        except: pass
    chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    dense_retriever = index.as_retriever(similarity_top_k=TOP_K)
    sparse_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)

    hybrid_retriever = QueryFusionRetriever([dense_retriever, sparse_retriever], similarity_top_k=TOP_K, mode="reciprocal_rerank", num_queries=FUSION_NUM_QUERIES)  # type: ignore
    return hybrid_retriever

def query_system(query: str, hybrid_retriever: QueryFusionRetriever) -> Tuple[str, list]:
    _ensure_api_key()
    retrieved_nodes = hybrid_retriever.retrieve(query)[:TOP_K]
    context = "\n\n".join(f"[Source: {n.node.metadata.get('Company_Name', '?')} ({n.node.metadata.get('Ticker', '?')}, {n.node.metadata.get('Quarter', '?')})]\n{n.node.get_content()}" for n in retrieved_nodes)
    prompt = STRICT_PROMPT.format(context=context, query=query)
    response = Settings.llm.complete(prompt)
    return str(response), retrieved_nodes