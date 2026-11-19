# Yotta Workstream A — Production-Grade RAG Pipeline

A robust, enterprise-grade Retrieval-Augmented Generation (RAG) system built with **Python 3.12** and **Streamlit** for the Yotta Senior AI Engineer take-home assessment.

This project implements an end-to-end pipeline connecting document ingestion, chunking, dense/BM25/hybrid indexing, two-pass relevance grading with continuation chunk support, bounded query rewriting with retry, and strictly grounded generation with verified source citations.

---

## 1. System Architecture & Information Flow

### Architecture Flowchart

```
                            ┌────────────────────────────────────────┐
                            │           User Query Input             │
                            └──────────────────┬─────────────────────┘
                                               │
                                               ▼
                     ┌──────────────────────────────────────────────────────┐
                     │                 Retrieval Stage                      │
                     │  • Dense Vector: Cosine Similarity (all-MiniLM-L6-v2) │
                     │  • Lexical Search: BM25Okapi (rank-bm25)             │
                     │  • Hybrid Fusion: Reciprocal Rank Fusion (RRF)       │
                     └─────────────────────────┬────────────────────────────┘
                                               │ Top-K Candidates
                                               ▼
                     ┌──────────────────────────────────────────────────────┐
                     │          Two-Pass Relevance Grading Stage            │
                     │  • Pass 1: Identify Primary Anchors (score ≥ 0.50)   │
                     │  • Pass 2: Sibling Continuation Chunks (score ≥ 0.35)│
                     │  • Strategy-Aware: Normalized BM25 & Hybrid Signals  │
                     └─────────────────────────┬────────────────────────────┘
                                               │
                         ┌─────────────────────┴──────────────────────┐
                         │                                            │
                 Context Insufficient                         Context Sufficient
                         │                                            │
                         ▼                                            │
        ┌──────────────────────────────────┐                          │
        │      Bounded Retry Rewrite       │                          │
        │  • Max 1 Query Reformulation     │                          │
        │  • Re-query & Re-grade           │                          │
        └────────────────┬─────────────────┘                          │
                         │                                            │
               Still Insufficient?                                    │
              ┌──────────┴──────────┐                                 │
             YES                    NO                                │
              │                      │                                │
              ▼                      └────────────────►───────────────┤
    ┌──────────────────────┐                                          ▼
    │   Strict System      │                        ┌───────────────────────────────────┐
    │     Abstention       │                        │   Grounded Generation Stage       │
    │  "Cannot answer..."  │                        │  • Strict prompt grounding        │
    └──────────────────────┘                        │  • Consolidated synthesis         │
                                                    │  • Extracted bracketed citations  │
                                                    │  • Hallucination filter against   │
                                                    │    active context passages        │
                                                    └─────────────────┬─────────────────┘
                                                                      │
                                                                      ▼
                                                    ┌───────────────────────────────────┐
                                                    │  Answer + Citations [doc_001]     │
                                                    │  Latency & Diagnostic Metadata    │
                                                    └───────────────────────────────────┘
```

---

## 2. Requirement Traceability Matrix

| Requirement / Scope Item | Component | Status | Implementation Details |
|---|---|:---:|---|
| **Reproducible MS MARCO subset** | Ingestion | **Completed** | Hugging Face MS MARCO (`microsoft/ms_marco` v1.1) with deterministic slicing |
| **Local JSONL fallback** | Ingestion | **Completed** | 20 curated technical passages in `data/sample_passages.jsonl` |
| **Passage cleaning & stable IDs** | Preprocessing | **Completed** | Unicode NFKC normalization, whitespace cleanup, SHA-256 deduplication, stable `doc_id` |
| **Fixed-size chunking with overlap** | Chunking | **Completed** | 300 character chunks with 50 character overlap and word-boundary snapping |
| **Sentence-boundary chunking** | Chunking | **Completed** | Structural chunking grouping complete grammatical sentences |
| **Dense vector retrieval** | Retrieval | **Completed** | NumPy in-memory dot product over $L_2$-normalized `all-MiniLM-L6-v2` embeddings on CPU |
| **BM25 lexical retrieval** | Retrieval | **Completed** | `rank-bm25` (BM25Okapi) index over identical tokenized chunks |
| **Hybrid retrieval (RRF)** | Retrieval | **Completed** | Reciprocal Rank Fusion combining Dense & BM25 rankings: $\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}(d)}$ |
| **Recall@K evaluation metric** | Evaluation | **Completed** | Parent `doc_id` deduplication preventing multiple chunks from inflating recall |
| **Two-pass relevance grading** | Pipeline | **Completed** | Pass 1 anchor threshold ($\ge 0.50$); Pass 2 continuation threshold ($\ge 0.35$); BM25 coverage normalization |
| **Query rewrite with single retry** | Pipeline | **Completed** | Strictly bounded ($\le 1$ retry loop); abstains gracefully if evidence remains insufficient |
| **Grounded generation** | Generation | **Completed** | Strict prompt grounding, consolidated synthesis, citation filtering against active context |
| **Streamlit interactive UI** | Presentation | **Completed** | Multi-tab UI (`app.py`) with query runner, diagnostics, and evaluation dashboard |
| **NVIDIA NIM provider support** | Providers | **Completed** | OpenAI-compatible endpoint with health checks and Nemotron model support |
| **Local CPU embedding baseline** | Providers | **Completed** | Local `all-MiniLM-L6-v2` with offline caching; zero cost, no GPU required |
| **Automated test suite** | Reliability | **Completed** | 66 passed unit and integration tests across all modules |

---

## 3. Setup and Run Instructions

### 1. Prerequisites
- Python 3.12 (or 3.10+)
- PowerShell (Windows) or Bash (macOS/Linux)
- Git

### 2. Environment Setup
```powershell
# Navigate into the project folder
cd yotta-rag-pipeline

# (Optional) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create .env from template
Copy-Item .env.example .env
```

### 3. Launching the Streamlit Web Application
```powershell
# Run the interactive Streamlit app
streamlit run app.py
```
Open your browser to `http://localhost:8501`. The application includes:
- **💬 Question Answering & Verification Tab:** Run custom questions using Dense, BM25, or Hybrid retrieval; configure Top-K, grading methods, and relevance thresholds; view consolidated answers, citation pills, latency, and expandable candidate chunk diagnostics.
- **📊 Retrieval Benchmark & Evaluation Tab:** Run Recall@1/3/5 benchmarks on `data/eval_queries.json`, inspect bar charts and per-query comparisons, and download `results/evaluation_results.csv`.
- **🏗️ Architecture & Diagnostics Tab:** In-app technical documentation and pipeline blueprints.

### 4. Running the End-to-End Verification CLI
```powershell
# Automated offline verification on supported and unsupported queries across all strategies
python scripts/manual_rag_test.py --mock-llm
```

### 5. Running the Retrieval Benchmark CLI
```powershell
# Executes Recall@1/3/5 benchmark and exports results/evaluation_results.csv
python rag/evaluation.py
```

### 6. Running the Automated Test Suite
```powershell
# Run the complete unit and integration test suite
python -m pytest tests/ -v
```

---

## 4. Provider Configuration

Edit `.env` to configure your LLM and embedding providers:

### Option A: Local Zero-Cost Offline Setup (Default / CI / Testing)
```env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Option B: NVIDIA NIM (Production Recommended)
```env
LLM_PROVIDER=nvidia_nim
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_API_KEY=nvapi-your-key-here
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Option C: OpenAI API
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Option D: Local Ollama (Llama 3 / Mistral)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 5. Architectural Design Decisions

### 1. Two-Pass Continuation Chunk Relevance Grading
- **The Problem:** In chunked corpora, a single parent document may contain a complete explanation split across chunks (e.g. `doc_001` split into `doc_001_f0` and `doc_001_f1`). Standalone vector scoring of the second chunk (`doc_001_f1`) may yield a lower similarity score (e.g. 0.4004) because query keywords were isolated in the first chunk, causing the grader to reject complementary evidence at threshold 0.50.
- **The Solution:** Two-pass evaluation:
  - **Pass 1:** Chunks meeting the primary threshold ($\ge 0.50$) confirm their parent document as an **anchor document** (`anchor_doc_ids`).
  - **Pass 2:** Sibling chunks from confirmed anchor documents are admitted if they meet a relaxed **continuation threshold**:
    $$\text{continuation\_threshold} = \max(0.25, \text{round}(\text{threshold} \times 0.70, 4)) \quad (\text{0.35 at default 0.50})$$
  - **Irrelevant Query Safety:** Unanchored documents (e.g. `doc_011` for unsupported queries) have no chunks meeting the primary threshold. Anchor set remains empty; therefore, continuation is never unlocked and all chunks are strictly rejected.

### 2. Strategy-Aware Score Normalization
- Comparing unbounded BM25 scores directly against cosine thresholds causes false rejections or false acceptances.
- We implemented `compute_bm25_relevance()` in `rag/grading.py`, calculating non-stopword content-term coverage and scaling scores into $[0.0, 1.0]$.
- Hybrid grading verifies underlying dense similarity and lexical coverage signals, preventing rank-only reciprocal values from admitting irrelevant passages.

### 3. Strictly Bounded Query Rewriting
- If initial retrieval yields no chunks passing grading, the pipeline reformulates the query using the LLM (`rewrite_query()`) and executes a second retrieval pass.
- **Strictly Bounded:** At most **1 retry** is attempted. If evidence remains insufficient, the pipeline abstains immediately without hallucinating.

### 4. Grounded Generation & Citation Verification
- Prompts enforce strict grounding: answering only from supplied context passages.
- Answers cite parent document IDs in brackets: `[doc_0001]`.
- The generator verifies every bracketed citation against the active context, discarding any hallucinated IDs that do not exist in the retrieved passages.

---

## 6. Sample Inputs, Test Data & Benchmark Results

### 1. Curated Corpus (`data/sample_passages.jsonl`)
Contains 20 verified technical passages covering diverse computer science domains:
- Supervised & Unsupervised Machine Learning (`doc_001`)
- Overfitting and Regularization (`doc_002`)
- Gradient Descent Optimization & Learning Rates (`doc_003`)
- Transformer Self-Attention Mechanics (`doc_004`)
- RAG Architecture & Hallucination Mitigation (`doc_005`)
- Lexical Search & BM25 Scoring (`doc_006`)
- Dense Semantic Vector Retrieval (`doc_007`)
- B-Tree Database Storage Engines (`doc_008`)
- Python Global Interpreter Lock (GIL) (`doc_009`)
- TLS Handshake & Cryptographic Exchanges (`doc_010`)
- Cloud Deployment Models (IaaS, PaaS, SaaS) (`doc_011`)
- Quantum Computing & Qubits (`doc_012`)
- K-Fold Cross-Validation (`doc_013`)
- Relational Database Normalization (`doc_014`)
- Docker Containerization vs Virtual Machines (`doc_015`)
- Cosine Similarity in Inner Product Spaces (`doc_016`)
- Raft Consensus Algorithm (`doc_017`)
- REST APIs vs gRPC Protocol Buffers (`doc_018`)
- Asymmetric Public-Key Cryptography (RSA) (`doc_019`)
- Zero-Shot Learning via Semantic Embedding Spaces (`doc_020`)

### 2. Labelled Test Set (`data/eval_queries.json`)
Contains 18 structured evaluation queries: **15 supported technical queries** mapped to ground-truth parent document IDs, and **3 negative out-of-domain error test queries** designed to verify system error handling, zero-evidence detection, and graceful abstention:
```json
// Supported Benchmark Query Example
{
  "query_id": "q_001",
  "query": "What is the difference between supervised and unsupervised learning?",
  "relevant_doc_ids": ["doc_001"],
  "category": "machine_learning"
}

// Negative / Error Verification Query Example (Topic Absent from Corpus)
{
  "query_id": "q_016",
  "query": "What is the current market share of Yotta in the Indian cloud computing market?",
  "relevant_doc_ids": [],
  "category": "unsupported",
  "is_supported": false,
  "error_type": "no_relevant_passage_retrieved",
  "description": "Negative test query: Tests pipeline abstention when question topic is completely absent from corpus."
}
```

The 3 negative / out-of-domain evaluation queries are:
1. `q_016`: *"What is the current market share of Yotta in the Indian cloud computing market?"* (Cloud business topic absent from corpus)
2. `q_017`: *"What are the primary symptoms and diagnostic biomarkers of Parkinson's disease?"* (Medical clinical domain absent from corpus)
3. `q_018`: *"Who won the FIFA World Cup final in 2022 and what was the final score?"* (Sports history absent from corpus)

### 3. Retrieval Evaluation Benchmark Results (`results/evaluation_results.csv`)

| Metric | Dense Retrieval (Cosine) | BM25 Retrieval (Okapi) | Winner / Analysis |
|---|:---:|:---:|---|
| **Mean Recall@1** | **100.00%** | **86.67%** | Dense captures semantic intent; BM25 missed 2 queries with lexical mismatch |
| **Mean Recall@3** | **100.00%** | **100.00%** | Both achieve complete recall at top-3 across all supported queries |
| **Mean Recall@5** | **100.00%** | **100.00%** | Both maintain complete recall at top-5 across all supported queries |

The complete query-by-query breakdown across all 18 queries is exported to `results/evaluation_results.csv`.

---

## 7. Failure Analysis & Error Mitigation Matrix

The table below outlines the core failure modes, root causes, and production mitigations implemented across the pipeline:

| Failure Mode | Root Cause | Pipeline Mitigation & Error Handling |
|---|---|---|
| **Dataset download fails** | Network failure, Hugging Face dataset revision deprecation, or library issue | **Local JSONL fallback:** When remote loading fails or times out, the system automatically switches to `data/sample_passages.jsonl` with deterministic first-N selection. |
| **No relevant passage retrieved** | Weak chunking, embedding dimension mismatch, poor query formulation, or small corpus | **Hybrid Retrieval + Bounded Rewrite:** Fuses dense semantic search and BM25Okapi rankings via RRF; triggers an automated LLM query reformulation when initial evidence is missing. |
| **False relevance grade** | Threshold poorly calibrated or grader prompt weak | **Calibrated Two-Pass Grading:** Pass 1 anchor threshold ($\ge 0.50$) unlocks sibling continuation chunks ($\ge 0.35$); BM25 scores normalized by non-stopword query term coverage. |
| **Hallucinated answer** | Prompt not strict or context does not support factual claim | **Strict Grounding & Citation Validation:** Prompt directs model to answer *only* from context; regex extractor cross-references bracketed `[doc_id]` citations against supplied passages and strips invalid IDs. |
| **Provider unavailable** | Missing API key, incorrect base URL, timeout, or model not deployed | **Startup Health Checks & Graceful Fallbacks:** Validates credentials and endpoint status via `health_check()`; provides automatic fallback to deterministic verification mock for offline environments. |
| **Retry does not help** | Rewrite changes semantic meaning or retrieves the same irrelevant passages | **Strictly Bounded Retry (Max 1):** Limits retries to exactly 1 query reformulation; logs both original and rewritten queries; immediately **abstains cleanly** if evidence remains insufficient. |

---

## 8. Automated Test Suite Summary

The repository includes 66 automated tests verifying all pipeline stages:
- `tests/test_pipeline.py` (16 tests): End-to-end question answering, irrelevant query retry and abstention, bounded retry limits, citation extraction and filtering, continuation chunk grading, unanchored rejection.
- `tests/test_evaluation.py` (6 tests): Recall@K computation, parent document deduplication, CSV export.
- `tests/test_retrieval.py` (6 tests): Dense, BM25, and Hybrid RRF ranking and edge cases.
- `tests/test_indexing.py` (7 tests): Matrix normalization, batch embeddings, BM25 tokenization.
- `tests/test_chunking.py` (8 tests): Fixed-size and sentence-boundary chunking.
- `tests/test_ingestion.py` (6 tests): Cleaning, deduplication, JSONL fallback.
- `tests/test_providers.py` (13 tests): Provider adapters, health checks, error sanitization.
- `tests/test_config.py` (4 tests): Typed environment configuration and secret masking.
