"""
app.py — Alpha-Detective Streamlit Web App
==========================================
A hybrid-search RAG chatbot for earnings call transcripts.
"""

from __future__ import annotations
import time
import streamlit as st
from alpha_detective import RetrievalConfig, build_hybrid_retriever, load_data, query_system

APP_RETRIEVAL_CONFIG = RetrievalConfig(
    retrieve_k=5,
    fused_k=5,
    final_k=5,
)

st.set_page_config(page_title="Alpha-Detective 🕵️", page_icon="🕵️", layout="wide")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.title("🕵️ Alpha-Detective")
    st.caption("Hybrid Search RAG — Earnings Call Analysis")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", 
                                   help="Get your key at platform.openai.com. Your key is never stored.")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop() # This stops the app from running until they paste a key

    col_a, col_b = st.columns(2)
    col_a.metric("Docs loaded", "✅" if st.session_state.data_loaded else "⏳")
    col_b.metric("Retriever", "RRF 📡" if st.session_state.retriever else "—")

    # Accept multiple CSV or TXT files
    uploaded_files = st.file_uploader("Upload transcripts (CSV or TXT)", type=["csv", "txt"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Load & Index Uploaded Data", type="primary", use_container_width=True):
            with st.spinner(f"⏳ Loading {len(uploaded_files)} files and building the hybrid index..."):
                try:
                    t0 = time.time()
                    docs = load_data(uploaded_files)
                    
                    st.session_state.retriever = build_hybrid_retriever(
                        docs,
                        api_key=openai_api_key,
                        config=APP_RETRIEVAL_CONFIG,
                    )
                    st.session_state.data_loaded = True
                    st.success(f"✅ Indexed {len(docs):,} chunks in {time.time() - t0:.1f}s. Hybrid retriever is live.")
                except Exception as exc:
                    st.error(f"❌ Unexpected error while indexing: {exc}")
    else:
        st.info("Please upload one or more files to begin.")

st.subheader("🔍 Question an earnings call")
st.caption("Examples: “What did Accenture say about AI bookings?” · “Revenue growth for Apple?”")

query = st.text_input("Your financial query", placeholder="Ask anything about the loaded earnings calls...", label_visibility="collapsed")

if query:
    if not st.session_state.data_loaded:
        st.warning("⚠️ No index yet. Upload files and click “Load & Index Uploaded Data” first.")
    else:
        try:
            with st.spinner("🕵️ Investigating..."):
                t0 = time.time()
                retriever = st.session_state.retriever
                if retriever is None:
                    raise RuntimeError("Load and index transcripts before querying.")
                answer, nodes = query_system(
                    query,
                    retriever,
                    api_key=openai_api_key,
                    config=APP_RETRIEVAL_CONFIG,
                )
            elapsed = time.time() - t0

            if retriever.rerank_warning:
                st.warning(retriever.rerank_warning)

            st.markdown("### 🧠 Answer")
            with st.container(border=True):
                st.markdown(answer)
            st.caption(f"Generated in {elapsed:.1f}s · top-{len(nodes)} fused sources")

            with st.expander("📚 Retrieved Context Sources", expanded=False):
                st.caption("Score = **RRF score**. Higher is better.")
                if nodes:
                    rows = [
                        {
                            "Company Name": n.node.metadata.get("Company_Name", "?"),
                            "Ticker": n.node.metadata.get("Ticker", "?"),
                            "Quarter": n.node.metadata.get("Quarter", "?"),
                            "RRF Score": n.node.metadata.get("rrf_score", n.score),
                            "Rerank Score": n.node.metadata.get("rerank_score"),
                        }
                        for n in nodes
                    ]
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    with st.expander("Show source snippets"):
                        for i, n in enumerate(nodes, 1):
                            st.markdown(f"**{i}.** {n.node.metadata.get('Company_Name', '?')} · {n.node.metadata.get('Ticker', '?')} · {n.node.metadata.get('Quarter', '?')}")
                            snippet = n.node.get_content()[:400].replace("\n", " ")
                            st.markdown(f"> {snippet}…")
                else:
                    st.markdown("*No sources retrieved.*")
        except Exception as exc:
            st.error(f"❌ {exc}")