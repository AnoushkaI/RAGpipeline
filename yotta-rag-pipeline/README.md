# Yotta Workstream A — RAG Pipeline

A production-grade, modular Retrieval-Augmented Generation (RAG) prototype built with **Python 3.12** and **Streamlit** for the Yotta Senior AI Engineer take-home assignment.

---

## 1. What This System Does (Plain-Language Architecture)

Large Language Models (LLMs) often hallucinate or answer from outdated pretrained knowledge when asked specific questions. Retrieval-Augmented Generation solves this by:
1. **Searching a targeted collection of documents** for verified factual passages relevant to the user's question using Dense vector similarity, BM25 lexical keyword matching, or Hybrid Reciprocal Rank Fusion (RRF).
2. **Grading the retrieved evidence** (using configurable threshold or LLM semantic evaluation) to ensure it actually contains the facts needed to answer.
3. **Rewriting and retrying retrieval once** if the initial evidence is insufficient or off-topic (strictly bounded to at most 1 retry).
4. **Instructing the model to answer strictly from the retrieved evidence** with bracketed citations to source document IDs (e.g. `[doc_0001]`), abstaining if the answer cannot be proven.

### End-to-End Information Flow
```
User Question
     │
     ▼
[Retriever: Dense Vector / BM25 Lexical / Hybrid RRF]
     │
     ▼
[Relevance Grader: Threshold or LLM Grader]
     │
     ├── Context Insufficient? ──► [Query Rewriter] ──► [Single Retry Retrieval]
     │                                                            │
     └── Context Sufficient? ◄────────────────────────────────────┘
     │
     ▼
[Grounded Answer Generator] (Cites Parent Passage IDs, strict anti-hallucination)
     │
     ▼
User Answer + Verified Source Citations + Quality & Latency Metrics
```

---

## 2. Requirement Traceability Matrix

| Requirement / Scope Item | Category | Status | Notes |
|---|---|---|---|
| **Reproducible MS MARCO subset** | Ingestion | **Implemented (Phase 2)** | Hugging Face MS MARCO (`microsoft/ms_marco` v1.1) with deterministic slicing |
| **Local JSONL fallback** | Ingestion | **Implemented (Phase 2)** | 20 curated technical passages in `data/sample_passages.jsonl` |
| **Passage cleaning & stable IDs** | Preprocessing | **Implemented (Phase 2)** | Unicode NFKC, control char removal, SHA-256 deduplication, stable `doc_id` |
| **Fixed-size chunking with overlap** | Chunking | **Implemented (Phase 2)** | Configurable `chunk_size` & `chunk_overlap` with word boundary snapping |
| **Sentence-boundary chunking** | Chunking | **Implemented (Phase 2)** | Alternative structural strategy grouping complete grammatical sentences |
| **Labelled evaluation queries** | Evaluation | **Implemented (Phase 2)** | 15 test queries in `data/eval_queries.json` mapped to relevant `doc_id`s |
| **Dense vector retrieval** | Retrieval | **Implemented (Phase 3)** | Cosine similarity via NumPy in-memory matrix over $L_2$-normalized vectors |
| **BM25 lexical retrieval** | Retrieval | **Implemented (Phase 3)** | `rank-bm25` (BM25Okapi) index over the exact same chunks |
| **Hybrid retrieval (RRF)** | Retrieval | **Implemented (Phase 4)** | Reciprocal Rank Fusion combining Dense & BM25, normalized to $[0.0, 1.0]$ |
| **Recall@K evaluation metric** | Evaluation | **Implemented (Phase 3)** | Strict parent `doc_id` mapping, deduplication preventing chunk inflation |
| **Relevance grading** | Pipeline | **Implemented (Phase 4)** | Configurable `threshold` or `llm` binary grading modes with per-chunk verdicts |
| **Query rewrite with single retry** | Pipeline | **Implemented (Phase 4)** | Bounded retry (strictly $\le 1$ retry loop) when context is insufficient |
| **Grounded answer generation** | Generation | **Implemented (Phase 4)** | Strict prompt grounding, single consolidated answer, citation extraction |
| **OpenAI provider support** | Providers | **Implemented (Phase 1)** | Fully configured with missing-key handling and safe masking |
| **NVIDIA NIM provider support** | Providers | **Implemented (Phase 1)** | OpenAI-compatible endpoint with live health checks and Nemotron model support |
| **Llama (Ollama) provider support** | Providers | **Implemented (Phase 1)** | REST API adapter with batched `/api/embed` and model presence checks |
| **Sentence-Transformers provider** | Providers | **Implemented (Phase 1)** | Local CPU embedding baseline (`all-MiniLM-L6-v2`) with fast offline caching |
| **Faithfulness evaluation rubric** | Evaluation | Pending (Phase 5) | Claim-by-claim supported/unsupported rubric |
| **Latency measurement** | Evaluation | **Implemented (Phase 4)** | Monotonic timer per pipeline query run |
| **Streamlit interactive UI** | Presentation | In Progress | App structure ready; interactive controls completed in Phase 5 |
| **Automated test suite** | Reliability | **Implemented (Phases 1-4)** | 60 passed unit and integration tests across all modules |

*Statuses: `Implemented`, `In Progress`, `Pending`, `Untested`.*

---

## 3. Project Structure

```
yotta-rag-pipeline/
├── app.py                      # Streamlit UI application
├── config.py                   # Environment-driven typed configuration & safe masking
├── requirements.txt            # Minimal dependencies (NumPy, rank-bm25; no vector DB / LangChain)
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules protecting secrets
├── README.md                   # Architecture, setup, and usage documentation
├── PROJECT_PROGRESS.md         # 5-phase handoff record and audit log
├── data/
│   ├── sample_passages.jsonl   # Curated local fallback dataset (20 passages)
│   └── eval_queries.json       # Labelled test set with parent passage IDs
├── rag/
│   ├── __init__.py             # Public API exports
│   ├── ingestion.py            # MS MARCO loader, cleaning, and deduplication
│   ├── chunking.py             # Fixed-size & sentence-boundary chunking
│   ├── providers.py            # ModelProvider, OpenAI, NIM, Ollama, Local ST
│   ├── indexing.py             # Dense matrix (NumPy) & BM25 indices
│   ├── retrieval.py            # Dense, BM25, and Hybrid (RRF) retrievers
│   ├── grading.py              # Threshold and LLM relevance grading
│   ├── generation.py           # Grounded answer generator with citation verification
│   ├── pipeline.py             # End-to-end orchestration with bounded 1-retry rewrite
│   └── evaluation.py           # Recall@K calculation and benchmark suite
├── scripts/
│   └── manual_retrieval_test.py # Interactive CLI tool for testing retrieval
├── tests/
│   ├── test_config.py          # Configuration and masking tests
│   ├── test_providers.py       # Provider adapters and error handling tests
│   ├── test_ingestion.py       # Dataset loading, cleaning, and deduplication tests
│   ├── test_chunking.py        # Chunking strategy tests
│   ├── test_indexing.py        # Dense & BM25 index creation tests
│   ├── test_retrieval.py       # Retrieval algorithms and ranking tests
│   ├── test_evaluation.py      # Benchmark and Recall@K calculation tests
│   └── test_pipeline.py       # End-to-end pipeline integration tests
└── results/
    └── .gitkeep                # Output folder for evaluation metrics
```

---

## 4. Setup and Quick Start

### 1. Prerequisites
- Python 3.12 (or 3.10+)
- PowerShell (Windows) or Bash (macOS/Linux)

### 2. Environment Setup (PowerShell)
```powershell
# Navigate into the project folder
cd yotta-rag-pipeline

# (Optional) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create .env from template
Copy-Item .env.example .env
```

### 3. Provider Configuration
Open `.env` in an editor and select your providers:

- **Option A: Local Zero-Cost Setup (No API Keys Required)**
  ```env
  LLM_PROVIDER=mock
  EMBEDDING_PROVIDER=sentence_transformers
  EMBEDDING_MODEL=all-MiniLM-L6-v2
  ```

- **Option B: NVIDIA NIM (Nemotron / Llama)**
  ```env
  LLM_PROVIDER=nvidia_nim
  NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
  NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b
  NVIDIA_API_KEY=nvapi-...
  EMBEDDING_PROVIDER=sentence_transformers
  ```

- **Option C: OpenAI API**
  ```env
  LLM_PROVIDER=openai
  LLM_MODEL=gpt-4o-mini
  OPENAI_API_KEY=sk-...
  EMBEDDING_PROVIDER=sentence_transformers
  ```

---

## 5. Running the Pipeline (Phase 4 Python API)

```python
from rag.pipeline import create_pipeline

# 1. Initialize pipeline and auto-index the local 20-passage corpus
pipeline = create_pipeline()

# 2. Query using dense, bm25, or hybrid retrieval
response = pipeline.run(
    query="How does backpropagation compute gradients?",
    strategy="hybrid",  # 'dense' | 'bm25' | 'hybrid'
    top_k=3,
)

print(f"Answer: {response.answer}")
print(f"Cited Documents: {response.cited_doc_ids}")
print(f"Latency: {response.latency_ms:.1f}ms")
print(f"Retry Triggered: {response.retry_attempted}")
```

### Interactive Manual Retrieval Test CLI
```powershell
python scripts/manual_retrieval_test.py
```

### Retrieval Evaluation Benchmark
```powershell
python rag/evaluation.py
```

### Automated Test Suite
```powershell
python -m pytest tests/ -v
```

---

## 6. Phase 4 Architecture & Design Decisions

### Relevance Grading
- **Threshold Grader:** Compares raw similarity scores to `relevance_threshold` (default 0.5). Dense cosine similarities and normalized RRF scores map to $[0, 1]$.
- **LLM Grader:** Zero-shot binary classifier asking the LLM to output `YES` or `NO`. If the LLM call fails or times out, the grader falls back to threshold grading.

### Bounded Query Rewriting
- If initial retrieval yields no chunks passing grading, the pipeline reformulates the query using the LLM and runs a second retrieval pass.
- **Strictly Bounded:** At most **1 retry** is attempted. If the second attempt is also insufficient, the pipeline abstains immediately without hallucinating.

### Citation Attribution
- Answers cite parent document IDs in brackets: `[doc_0001]`.
- The generator verifies each citation against the active context, stripping out any hallucinated IDs that do not exist in the retrieved passages.
