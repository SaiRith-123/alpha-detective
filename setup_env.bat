@echo off
cd /d C:\Users\srisa\.openclaw-autoclaw\workspace\alpha-detective
REM Prefer py -3.12 (matches user's Python 3.12); fall back to `python`.
py -3.12 -m venv .venv || python -m venv .venv
set PY=.venv\Scripts\python.exe
"%PY%" -m pip install --upgrade pip setuptools wheel
"%PY%" -m pip install "streamlit" "pandas" "chromadb" "llama-index" "llama-index-vector-stores-chroma" "rank-bm25" "llama-index-retrievers-bm25" "llama-index-llms-openai" "llama-index-embeddings-openai"
echo === INSTALL_DONE ===
"%PY%" -c "import sys; print('PY', sys.version.split()[0]); import llama_index as li, chromadb, streamlit, pandas, rank_bm25; print('llama_index', li.__version__); print('chromadb', chromadb.__version__); print('streamlit', streamlit.__version__); print('pandas', pandas.__version__); print('rank_bm25 ok')"