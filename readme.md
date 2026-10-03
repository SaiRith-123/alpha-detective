# 🕵️ Alpha-Detective

### Hybrid Search RAG Pipeline for Earnings Call Analysis

> **An end-to-end Retrieval-Augmented Generation (RAG) application for analyzing corporate earnings call transcripts using hybrid semantic + lexical search, Reciprocal Rank Fusion, and source-grounded LLM responses.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange.svg)](https://www.llamaindex.ai/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-black.svg)](https://platform.openai.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![BM25](https://img.shields.io/badge/Retrieval-BM25-green.svg)]()

---

## 📌 Overview

**Alpha-Detective** is a prototype **Retrieval-Augmented Generation (RAG)** application for querying corporate earnings call transcripts.

Instead of relying exclusively on semantic vector search, Alpha-Detective combines two complementary retrieval strategies:

* 🧠 **Dense Retrieval** — OpenAI embeddings + ChromaDB
* 🔎 **Sparse Retrieval** — BM25 lexical search
* 🔀 **Reciprocal Rank Fusion (RRF)** — combines results from both retrieval systems
* 🤖 **LLM Generation** — produces answers grounded in retrieved transcript context
* 📚 **Source Attribution** — exposes the retrieved companies, tickers, and RRF scores used to generate each answer

This hybrid architecture allows the system to handle both **semantic questions** and **exact financial terminology, names, and keywords** more effectively than relying on a single retrieval method.

*DATA SET: https://www.kaggle.com/datasets/rafifellert/2020-2026-transcripts-of-earning-calls*

---

# 🏗️ Architecture

```text
                  ┌─────────────────────────┐
                  │ Earnings Call Dataset   │
                  │ 1,185 Transcripts       │
                  │ 50 Companies            │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Data Preparation     │
                  │    combine_data.py      │
                  │                         │
                  │ • Dataset Detection     │
                  │ • CSV Generation        │
                  │ • Encoding Handling     │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Transcript Documents    │
                  │ Ticker / Company /      │
                  │ Quarter / Text          │
                  └────────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
      ┌─────────────────────┐      ┌─────────────────────┐
      │  Dense Retrieval    │      │  Sparse Retrieval   │
      │                     │      │                     │
      │ OpenAI Embeddings   │      │      BM25           │
      │        ↓            │      │        ↓            │
      │     ChromaDB        │      │ rank-bm25           │
      └──────────┬──────────┘      └──────────┬──────────┘
                 │                            │
                 └──────────────┬─────────────┘
                                ▼
                  ┌─────────────────────────┐
                  │ Reciprocal Rank Fusion  │
                  │                         │
                  │ LlamaIndex              │
                  │ QueryFusionRetriever    │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Retrieved Context       │
                  │                         │
                  │ Company                 │
                  │ Ticker                  │
                  │ RRF Score               │
                  │ Transcript Context      │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │      LLM Generation     │
                  │                         │
                  │      GPT-3.5-Turbo      │
                  │      Strict Prompting   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │     Streamlit App       │
                  │                         │
                  │ • Financial Q&A         │
                  │ • Grounded Answers      │
                  │ • Source Verification   │
                  │ • RRF Scores            │
                  └─────────────────────────┘
```

---

# 💡 Why Hybrid Search?

Traditional RAG systems often depend entirely on vector similarity.

While semantic search is powerful, financial documents contain many situations where **exact lexical matching matters**.

For example:

```text
"What did Accenture say about generative AI?"
```

A dense retriever may identify semantically related passages, while BM25 can strongly prioritize exact occurrences of terms such as:

```text
Accenture
generative AI
GenAI
AI revenue
```

Alpha-Detective combines both approaches.

### Dense Search

Uses OpenAI embeddings to identify semantically similar passages.

```text
Query
  ↓
Embedding
  ↓
ChromaDB
  ↓
Semantic Matches
```

### Sparse Search

Uses BM25 to identify documents based on lexical relevance.

```text
Query
  ↓
Tokenization
  ↓
BM25
  ↓
Keyword Matches
```

### Reciprocal Rank Fusion

The two ranked result sets are combined using **Reciprocal Rank Fusion (RRF)**.

```text
Dense Results ─────┐
                   ├──► RRF ──► Final Ranked Context
BM25 Results ──────┘
```

This allows the system to benefit from both **semantic similarity and exact keyword relevance**.

---

# 🔎 Source-Grounded Answers

Alpha-Detective is designed to make the retrieval process transparent.

Every response exposes the retrieved context sources:

| Company   | Ticker | RRF Score |
| --------- | ------ | --------: |
| Accenture | ACN    |    0.0321 |
| Microsoft | MSFT   |    0.0287 |
| NVIDIA    | NVDA   |    0.0249 |

This makes it possible to inspect **which documents influenced the answer** instead of treating the LLM as an unexplained black box.

> **The application separates retrieval from generation, making the RAG pipeline easier to inspect and debug.**

---

# 📊 Dataset

The project is designed around an earnings-call transcript dataset containing:

* **50 companies**
* **1,185 quarterly transcripts**
* Company names
* Stock tickers
* Fiscal quarters
* Full transcript text

The raw dataset is organized under:

```text
NLP_Dataset/
```

The optional preprocessing script converts the dataset into a consolidated CSV:

```text
earnings_transcripts.csv
```

with the schema:

```text
Ticker
Company_Name
Quarter
Text
```

---

# 📂 Project Structure

```text
alpha-detective/
│
├── NLP_Dataset/
│   └── ... earnings call transcripts
│
├── combine_data.py
│   └── Dataset consolidation and preprocessing
│
├── alpha_detective.py
│   └── Hybrid RAG engine
│
├── app.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── setup_env.bat
│   └── Windows environment setup
│
├── earnings_transcripts.csv
│   └── Consolidated transcript dataset
│
└── README.md
```

> Large datasets and generated vector indexes should generally be excluded from Git using `.gitignore`.

---

# 🛠️ Tech Stack

| Component        | Technology           | Purpose                     |
| ---------------- | -------------------- | --------------------------- |
| Language         | Python 3.12          | Application development     |
| RAG Framework    | LlamaIndex           | Retrieval orchestration     |
| Dense Retrieval  | OpenAI Embeddings    | Semantic search             |
| Vector Database  | ChromaDB             | Vector storage              |
| Sparse Retrieval | BM25                 | Keyword-based search        |
| Fusion           | QueryFusionRetriever | Hybrid retrieval            |
| LLM              | GPT-3.5-Turbo        | Answer generation           |
| Frontend         | Streamlit            | Interactive web application |
| Data Processing  | Python / Pandas      | Dataset preparation         |

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/alpha-detective.git

cd alpha-detective
```

---

## 2. Create a Virtual Environment

Alpha-Detective currently targets **Python 3.12**.

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## 3. Upgrade Packaging Tools

Upgrading the packaging tools first can help avoid dependency resolution and wheel installation issues.

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Prepare the Dataset

Dataset preparation is optional if you already have the consolidated transcript CSV.

If using the raw nested dataset:

```bash
python combine_data.py
```

The script automatically detects supported dataset layouts and generates:

```text
earnings_transcripts.csv
```

The preprocessing script also handles **Windows-1252 encoded characters**, including smart quotes commonly found in transcript files.

---

# 🔐 OpenAI API Key

Alpha-Detective follows a **Bring Your Own Key (BYOK)** approach.

The API key is entered through the Streamlit interface rather than hardcoded into source files.

### Why?

This prevents accidentally committing API credentials to GitHub.

**Never do this:**

```python
OPENAI_API_KEY = "sk-..."
```

Instead, enter your key through the application's secure input field.

You can obtain an API key from the [OpenAI Platform](https://platform.openai.com/api-keys?utm_source=chatgpt.com).

> ⚠️ Never commit API keys, `.env` files containing secrets, or credentials to GitHub.

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

Typical local address:

```text
http://localhost:8501
```

---

# 🖥️ Using the Application

### Step 1 — Enter API Key

Paste your OpenAI API key into the sidebar.

### Step 2 — Upload Transcripts

Upload one or more:

```text
.csv
.txt
```

files.

You can use:

```text
earnings_transcripts.csv
```

### Step 3 — Build the Index

Click:

```text
Load & Index Uploaded Data
```

The application will process the transcripts and generate embeddings.

> The first run may take longer because documents need to be embedded and indexed.

### Step 4 — Ask a Question

Example:

```text
What did Accenture say about generative AI?
```

Other examples:

```text
Which companies discussed AI-related revenue growth?

What were Microsoft's expectations for cloud growth?

Which companies mentioned margin pressure?

What challenges did NVIDIA discuss regarding supply?

How did management describe demand for generative AI?
```

### Step 5 — Inspect the Sources

The application displays:

* Retrieved company
* Ticker
* RRF score
* Retrieved context

This allows you to verify the documents used to construct the response.

---

# 🧠 RAG Pipeline

The complete question-answering flow is:

```text
User Question
      │
      ▼
Query Processing
      │
      ├───────────────┐
      ▼               ▼
 Dense Search      BM25 Search
      │               │
      ▼               ▼
 ChromaDB          rank-bm25
      │               │
      └───────┬───────┘
              ▼
        RRF Fusion
              │
              ▼
      Top Retrieved Context
              │
              ▼
       Strict Prompt
              │
              ▼
        GPT-3.5-Turbo
              │
              ▼
       Grounded Answer
              │
              ▼
      Retrieved Sources
```

---

# ⚡ Retrieval Configuration

The system uses LlamaIndex's `QueryFusionRetriever` to combine dense and sparse retrieval.

The configuration intentionally uses:

```text
num_queries = 1
```

This prevents additional query-expansion LLM calls.

The resulting architecture focuses on:

```text
Original Query
      ↓
Dense Retrieval
      +
Sparse Retrieval
      ↓
RRF
      ↓
Answer
```

rather than generating multiple rewritten queries.

---

# 💰 Cost Considerations

Alpha-Detective follows a **BYOK model**.

Users provide their own OpenAI API key, meaning:

* You don't pay for other users' API usage.
* Embedding generation consumes API credits.
* LLM responses consume API credits.
* Re-indexing a large dataset can generate additional embedding costs.

For experimentation and portfolio demonstrations, the overall API usage can generally be kept relatively small depending on dataset size and query volume.

---

# 💾 Persistence

The ChromaDB vector index is persisted locally:

```text
./chroma_db
```

Uploaded node content and metadata determine a fingerprinted Chroma collection. Re-indexing the same data reuses its stored vectors instead of embedding them again; changed data gets its own collection. The sparse BM25 retriever is rebuilt in memory from the uploaded nodes each time, since BM25 is not persisted.

To completely rebuild the index:

```text
Delete the chroma_db/ directory
```

and run the indexing process again.

---

# 📊 Retrieval Evaluation

Dataset: 1,185 transcript rows across 50 companies; human-reviewed questions: 0; embedding model: `text-embedding-3-small`. The current draft set is LLM-drafted. Automated screening can mark rows ready after exact-chunk validation, but that is not human review.

A preliminary `eval/results.csv` exists locally, but it contains only one machine-screened question and is not a meaningful retrieval comparison. The BGE reranker was unavailable during that run and fell back to the non-reranked path. No human-reviewed evaluation results or supported performance conclusions are available.

Configured ablation ladder:

| Configuration | Status |
|---|---|
| Dense-only, k=5 | Not measured |
| BM25-only, k=5 | Not measured |
| Hybrid, k=5 | Not measured |
| Hybrid wide fusion (20 → 20 → 6) | Not measured |
| Hybrid + filter | Not measured |
| Hybrid + filter + BGE rerank | Not measured |
| Hybrid + filter + FlashRank rerank | Not measured |

## Known limitations

* Aggregate questions may need evidence from multiple transcript chunks; the current exact-snippet scoring does not evaluate answer synthesis across chunks.
* There are no reviewed evaluation questions yet. Until a question set is assembled, no retrieval comparison can be made; even after that, small evaluation sets make small metric differences noisy.

Draft reviewable questions from the transcript dataset:

```text
python -m eval.make_golden_draft
```

To machine-screen drafts locally, set `OPENAI_API_KEY` in PowerShell and run:

```powershell
$env:OPENAI_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m eval.auto_review_golden --run-eval
```

This writes a separate `eval/golden.auto.jsonl`, preserving the original
`golden.jsonl`. It marks a row ready only when the LLM response passes checks
for metadata, type, long verbatim question overlap, exact evidence text, and
evidence occurring in exactly one transcript chunk. These checks cannot prove
that a question is factually well-posed. Automated screening is not equivalent
to hand review; inspect the generated set before treating it as a
human-validated benchmark. Rows that fail checks remain flagged for review.

The `--run-eval` option runs the dense-only, BM25-only, hybrid k=5 baseline,
wider-fusion, filtering, and reranking ablations using the machine-screened
output. To run evaluation separately:

```text
python -m eval.run_eval --golden eval/golden.auto.jsonl
```

Evaluation reads the key from the environment, reuses the fingerprinted Chroma
index when available, skips rows still requiring review, and writes measured
aggregate and per-type metrics to `eval/results.csv`. The harness includes
filtered hybrid runs with both the BGE and FlashRank rerankers.

Optional rerankers can be installed separately:

```text
pip install -r requirements-rerank.txt
```

This installs the BGE SentenceTransformer integration and the FlashRank CPU
ONNX backend. If a selected backend is not installed, the app warns and
continues without reranking.

---

# 🛡️ Error Handling & Security

The application includes safeguards for common runtime issues.

Examples include:

* Missing API key
* Missing dataset files
* Invalid uploaded files
* Missing folders
* Indexing failures

The application displays user-facing Streamlit errors instead of failing silently.

### Security practices

* API keys are entered at runtime
* Secrets are not hardcoded
* Credentials should never be committed to Git
* `.gitignore` should exclude local secrets and generated indexes

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc

.env
.env.*

chroma_db/

*.db

.DS_Store

.ipynb_checkpoints/
```

---

# 📸 Screenshots

Add screenshots here to showcase the application.

Recommended screenshots:

### Application Interface

```markdown
![Alpha-Detective UI](docs/app.png)
```

### Retrieved Sources

```markdown
![Retrieved Sources](docs/sources.png)
```

### Hybrid Retrieval Results

```markdown
![Hybrid Search](docs/retrieval.png)
```

A short GIF demonstrating:

```text
Question → Retrieval → Answer → Sources
```

would make the repository even stronger.

---

# 🔬 Key Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Dense vector retrieval
* Sparse lexical retrieval
* BM25 ranking
* Reciprocal Rank Fusion
* Vector databases
* Embedding-based semantic search
* LLM prompt engineering
* Source attribution
* Document preprocessing
* Persistent vector indexes
* Streamlit application development
* Secure API-key handling
* Hybrid information retrieval

---

# 🚧 Future Improvements

Potential extensions include:

* [ ] Upgrade to newer OpenAI models
* [ ] Add SHAP-style retrieval diagnostics
* [ ] Add reranking with a cross-encoder
* [ ] Add metadata filtering by company and quarter
* [ ] Add conversation memory
* [ ] Add financial metric extraction
* [ ] Add citation-level transcript references
* [ ] Add evaluation using RAGAS
* [ ] Add automated retrieval benchmarks
* [ ] Containerize with Docker
* [ ] Add CI/CD with GitHub Actions
* [ ] Deploy to a cloud platform
* [ ] Add observability and latency monitoring

---

# 📈 Potential Production Architecture

A production deployment could extend the current system into:

```text
                 ┌─────────────────┐
                 │ Transcript Data │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Data Pipeline   │
                 └────────┬────────┘
                          ▼
              ┌───────────────────────┐
              │ Document Processing   │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │ Hybrid Retrieval      │
              │                       │
              │ Vector + BM25 + RRF   │
              └───────────┬───────────┘
                          ▼
                 ┌─────────────────┐
                 │ Reranker        │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ LLM Generation  │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ API / Web App   │
                 └─────────────────┘
```

---

# 👨‍💻 Author

**Sai Rithesh Mandalapu**

Machine Learning & AI Enthusiast

---

## ⭐ If You Found This Project Useful

Consider giving the repository a ⭐ and exploring the implementation.

Built to demonstrate how **hybrid information retrieval can make financial-domain RAG systems more transparent and searchable. Generated answers should be checked against their cited transcript sources.**
