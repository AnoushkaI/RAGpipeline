# Yotta Workstream A — RAG Pipeline

A production-grade, modular Retrieval-Augmented Generation (RAG) prototype built with **Python 3.12** and **Streamlit** for the Yotta Senior AI Engineer take-home assignment.

---

## 1. What This System Does (Plain-Language Architecture)

Large Language Models (LLMs) often hallucinate or answer from outdated pretrained knowledge when asked specific questions. Retrieval-Augmented Generation solves this by:
1. **Searching a targeted collection of documents** for verified factual passages relevant to the user's question.
2. **Grading the retrieved evidence** to ensure it actually contains the facts needed to answer.
3. **Rewriting and retrying retrieval once** if the initial evidence is insufficient or off-topic.
4. **Instructing the model to answer strictly from the retrieved evidence** (with citations to source IDs), abstaining if the answer cannot be proven.

### End-to-End Information Flow
```
User Question
     │
     ▼
[Retriever: BM25 Lexical + Dense Cosine Similarity]
     │
     ▼
[Relevance Grader: Threshold or LLM Grader]
     │
     ├── Context Insufficient? ──► [Query Rewriter] ──► [Single Retry Retrieval]
     │                                                            │
     └── Context Sufficient? ◄────────────────────────────────────┘
     │
     ▼
[Grounded Answer Generator] (Cites Passage IDs, strict anti-hallucination)
     │
     ▼
User Answer + Source Citations + Quality & Latency Metrics
```

---

## 2. Requirement Traceability Matrix

| Requirement / Scope Item | Category | Status | Notes |
|---|---|---|---|
| **Reproducible MS MARCO subset** | Ingestion | **Implemented (Phase 2)** | Hugging Face MS MARCO with deterministic slicing & auto-fallback |
| **Local JSONL fallback** | Ingestion | **Implemented (Phase 2)** | 20 curated technical passages in `data/sample_passages.jsonl` |
| **Passage cleaning & stable IDs** | Preprocessing | **Implemented (Phase 2)** | Unicode NFKC, control char removal, SHA-256 deduplication, stable `doc_id` |
| **Fixed-size chunking with overlap** | Chunking | **Implemented (Phase 2)** | Configurable `chunk_size` & `chunk_overlap` with word boundary snapping |
| **Sentence-boundary chunking** | Chunking | **Implemented (Phase 2)** | Alternative structural strategy grouping complete grammatical sentences |
| **Labelled evaluation queries** | Evaluation | **Implemented (Phase 2)** | 15 test queries in `data/eval_queries.json` mapped to relevant `doc_id`s |
| **Dense vector retrieval** | Retrieval | Pending (Phase 3) | Cosine similarity via NumPy in-memory matrix |
| **BM25 lexical retrieval** | Retrieval | Pending (Phase 3) | `rank-bm25` baseline |
| **Recall@K evaluation metric** | Evaluation | Pending (Phase 3/5) | Labelled queries vs ground-truth passage IDs |
| **Relevance grading** | Pipeline | Pending (Phase 4) | Threshold or LLM-based context evaluation |
| **Query rewrite with single retry** | Pipeline | Pending (Phase 4) | Bounded retry (max 1 retry loop) |
| **Grounded answer generation** | Generation | Pending (Phase 4) | Strict prompt citations & abstention |
| **OpenAI provider support** | Providers | Implemented (Phase 1) | Fully configured with missing-key handling |
| **NVIDIA NIM provider support** | Providers | Implemented (Phase 1) | OpenAI-compatible endpoint support |
| **Llama (Ollama) provider support** | Providers | Implemented (Phase 1) | Local / remote REST API adapter |
| **Sentence-Transformers provider** | Providers | Implemented (Phase 1) | Zero-cost local dense embedding baseline |
| **Faithfulness evaluation rubric** | Evaluation | Pending (Phase 5) | Claim-by-claim supported/unsupported rubric |
| **Latency measurement** | Evaluation | Pending (Phase 5) | Monotonic timer per stage |
| **Streamlit interactive UI** | Presentation | In Progress (Phase 1 UI ready) | Full interactive demo completed in Phase 5 |
| **Unit tests & smoke tests** | Reliability | Implemented (Phase 1) | Automated pytest suite + provider smoke test |

*Statuses: `Implemented`, `In Progress`, `Pending`, `Untested`.*

---

## 3. Project Structure

```
yotta-rag-pipeline/
├── app.py                      # Streamlit UI application
├── config.py                   # Environment-driven typed configuration & safe masking
├── requirements.txt            # Minimal dependencies (no vector DB / LangChain)
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules protecting secrets
├── README.md                   # Architecture, setup, and usage documentation
├── PROJECT_PROGRESS.md         # 5-phase handoff record and audit log
├── data/
│   ├── .gitkeep
│   ├── sample_passages.jsonl   # (Phase 2) Local fallback dataset
│   └── eval_queries.json       # (Phase 2) Labelled test set with passage IDs
├── rag/
│   ├── __init__.py
│   ├── ingestion.py            # (Phase 2) MS MARCO loader and cleaner
│   ├── chunking.py             # (Phase 2) Fixed-size & sentence-boundary chunking
│   ├── providers.py            # (Phase 1) ModelProvider, OpenAI, NIM, Ollama, Local ST
│   ├── indexing.py             # (Phase 3) Dense matrix & BM25 indices
│   ├── retrieval.py            # (Phase 3) Dense & BM25 retrieval algorithms
│   ├── grading.py              # (Phase 4) Relevance grading logic
│   ├── generation.py           # (Phase 4) Grounded answer generator with citations
│   ├── pipeline.py             # (Phase 4) Orchestration with bounded 1-shot retry
│   └── evaluation.py           # (Phase 5) Recall@K, faithfulness, latency
├── tests/
│   ├── __init__.py
│   ├── test_config.py          # Phase 1 configuration and masking tests
│   └── test_providers.py       # Phase 1 provider adapters and error handling tests
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

- **Option B: OpenAI API**
  ```env
  LLM_PROVIDER=openai
  LLM_MODEL=gpt-4o-mini
  OPENAI_API_KEY=sk-...
  EMBEDDING_PROVIDER=sentence_transformers  # or 'openai' with text-embedding-3-small
  ```

- **Option C: NVIDIA NIM**
  ```env
  LLM_PROVIDER=nvidia_nim
  NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
  NVIDIA_MODEL=meta/llama-3.1-8b-instruct
  NVIDIA_API_KEY=nvapi-...
  ```

- **Option D: Local Llama via Ollama**
  ```env
  LLM_PROVIDER=ollama
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=llama3.2
  ```

### 4. Run the Provider Smoke Test
Test provider connectivity and credentials safely:
```powershell
python -m rag.providers
```

### 5. Run the Automated Tests
```powershell
pytest -v
```

### 6. Launch the Streamlit App
```powershell
streamlit run app.py
```

---

## 5. Dataset Ingestion, Cleaning & Fallback

### Dataset Specification & Selection Method
* **Dataset ID:** `microsoft/ms_marco`
* **Configuration:** `v1.1`
* **Split:** `train`
* **Selection Method:** Deterministic sequential first-$N$ slice (e.g., `train[:sample_size * 2]`). Sequential slicing is deliberately chosen over pseudo-random sampling to guarantee exact cross-environment reproducibility without seed drift.
* **Requested Subset Size:** Configurable via `sample_size` parameter (default: 100 passages).

### Cleaning & Deduplication Pipeline
1. **Unicode Normalization:** Applies NFKC normalization to standardize curly quotes, hyphens, and accents.
2. **Control Character Stripping:** Removes non-printable characters while preserving valid spaces, tabs, and newlines.
3. **Whitespace Compaction:** Collapses runs of consecutive spaces/tabs into a single space and caps excessive newlines at two.
4. **Quality Threshold:** Discards fragments and empty strings shorter than `min_length=20` characters.
5. **Deduplication:** Computes a SHA-256 hash of normalized lowercase text; duplicate content hashes are discarded.

> [!IMPORTANT]
> **Requested vs. Retained Passage Count:**  
> The number of raw passages requested (`sample_size`) is an upper target. Because the ingestion pipeline strictly filters low-quality stubs and eliminates duplicate passages, the **actual retained count is variable and will be $\le \text{sample\_size}$**. The pipeline does not claim a fixed final count.

### Local JSONL Fallback & Live Verification Status
* **Curated Local Fallback (`data/sample_passages.jsonl`):** Contains 20 curated, high-information passages spanning Machine Learning, Information Retrieval, Database Indexing, and Networking. **Verified and fully operational offline.**
* **Hugging Face Ingestion Status:** Verified live with `microsoft/ms_marco` (config `v1.1`, split `train`) with `fallback_on_error=False`, successfully extracting passages with provenance `source: hf_msmarco_v1.1`. When running in network-isolated environments or if Hugging Face is unreachable, the system automatically and transparently falls back to the curated local JSONL corpus.

---

## 6. Dense and BM25 Retrieval Architecture & Evaluation

### Retrieval Architecture
1. **Dense Vector Retrieval (`rag/indexing.py`, `rag/retrieval.py`):**
   * **Embedding Model:** Local `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional dense vectors).
   * **Vector Storage:** In-memory 2D NumPy array (`float32`), preserving chunk IDs, parent `doc_id` lineage, chunk text, and metadata.
   * **Safe Normalization:** Each row is $L_2$-normalized with an $\epsilon = 10^{-12}$ guardrail to prevent division-by-zero or `NaN` values.
   * **Cosine Similarity:** Computed via vectorized matrix-vector dot product (`np.dot(embeddings, query_vector)`). Handles zero vectors and dimension mismatches safely.
2. **Lexical Retrieval (BM25Okapi):**
   * **Algorithm:** `rank-bm25` (`BM25Okapi`) with standard saturation ($k_1=1.5$) and document length penalization ($b=0.75$).
   * **Tokenization:** Alphanumeric regex word tokenization (`\b\w+\b`) applied consistently at indexing and query time.
   * **Zero / Empty Handling:** Gracefully returns zero scores without raising exceptions on whitespace or empty queries.

### Fair Comparison Methodology
* **Unified Corpus & Chunks:** Both retrievers operate over the exact same 20 passages split into 40 chunks using the fixed-size strategy (`chunk_size=300`, `chunk_overlap=50`).
* **Lineage-Aware Recall@K:** Retrieved chunk IDs are mapped to parent document IDs before computing Recall@K. Taking the set of unique parent document IDs in the top-$K$ guarantees that **multiple chunks from the same relevant document are never counted more than once**.
* **Same Evaluation Benchmark:** Tested on 15 ground-truth queries in `data/eval_queries.json` across Machine Learning, Information Retrieval, Databases, Systems, and Security.

### Evaluation Commands
* **Run Retrieval Benchmark:**
  ```powershell
  python rag/evaluation.py
  # or
  python -m rag.evaluation
  ```
* **Run Phase 3 Tests:**
  ```powershell
  python -m pytest tests/test_indexing.py tests/test_retrieval.py tests/test_evaluation.py -v
  ```
* **Run Full Suite (49 tests):**
  ```powershell
  python -m pytest tests/ -v
  ```

### Actual Benchmark Results

```text
==================================================================
        RETRIEVAL EVALUATION REPORT: DENSE vs. BM25              
==================================================================
Evaluated Queries : 15
Evaluated K Levels: [1, 3, 5]
------------------------------------------------------------------
Metric             | Dense (Cosine)     | BM25 (Okapi)      
------------------------------------------------------------------
Mean Recall@1     |          100.00% |           86.67%
Mean Recall@3     |          100.00% |          100.00%
Mean Recall@5     |          100.00% |          100.00%
==================================================================
```

### Distinguishing Mocked Tests from Real Embedding Evaluation
* **Mocked Unit Tests:** Fast, deterministic unit tests use `MockEmbeddingProvider` and `ControlledEmbeddingProvider` to verify matrix dimensions, safe normalization, empty-input handling, zero vectors, dimension mismatch exceptions, and Recall@K math without touching the file system or PyTorch models.
* **Real Integration & Benchmark Evaluation:** `test_local_sentence_transformer_integration` and `rag/evaluation.py` run the actual `all-MiniLM-L6-v2` SentenceTransformer model on CPU using locally cached model weights.

### Performance & Hardware Profile
* **Target Hardware:** Windows laptop (8 GB RAM, RTX 2050 4 GB).
* **Inference Device:** Runs strictly on CPU (`device="cpu"`) with minimal memory footprint (< 500 MB peak RAM).
* **Offline Resolution:** Local cached snapshots are resolved via `_resolve_local_cache_path()` with `HF_HUB_OFFLINE=1`, eliminating external network requests.
* **Limitations:** The initial cold import of `transformers` / `torch` takes ~25-45 seconds on Windows; once loaded in-process, indexing 40 chunks takes ~1.1 seconds, and retrieval ranking takes < 5 milliseconds per query.

---

## 7. Security & Reliability Highlights
1. **No Secret Leaks:** All API keys are masked using `mask_secret()` before logging or rendering on the Streamlit dashboard.
2. **Graceful Degradation:** Missing credentials or unreachable server ports do not crash the application. Friendly diagnostics (`MISSING_CREDENTIALS` / `FAILED`) advise the user on next steps.
3. **No Heavy Framework Bloat:** Implemented directly with clean, readable Python, NumPy, and standard library constructs rather than complex agent frameworks or external vector databases.
