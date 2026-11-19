# PROJECT PROGRESS: Yotta Workstream A — RAG Pipeline

**Project:** Yotta Senior AI Engineer Take-Home • Workstream A: RAG Pipeline  
**Budget:** 180 minutes • 5 Phases  
**Current Phase:** Phase 3 Complete (Awaiting User Approval for Phase 4)  
**Last Updated:** Phase 3 Completion  

---

## Phase Status Summary

| Phase | Description | Allocated | Status | Gate Check / Deliverables |
|---|---|---|---|---|
| **Phase 1** | **Project Foundation & Provider Setup** | **25 min** | **COMPLETED & HARDENED** | Modular structure, provider abstraction, health checks, config & tests |
| **Phase 2** | **MS MARCO Ingestion, Cleaning & Chunking** | **40 min** | **COMPLETED** | Reproducible subset, local fallback JSONL, 2 chunking strategies, eval dataset |
| **Phase 3** | **Dense & BM25 Indexing / Retrieval** | **45 min** | **COMPLETED** | In-memory NumPy cosine similarity, BM25 index, fair Recall@K benchmark |
| **Phase 4** | Relevance Grading, Rewrite & Generation | 35 min | Pending Approval | Threshold/LLM grader, 1 bounded retry, grounded citation prompt |
| **Phase 5** | Streamlit UI, Tests, Evaluation & Docs | 35 min | Pending Phase 4 | Interactive UI, failure analysis, latency profile, full report |

---

## Phase 2 Execution Record: Ingestion, Cleaning & Chunking

### 1. Ingestion & Preprocessing Architecture (`rag/ingestion.py`)
- [x] **Passage Data Structure:** Implemented `Passage` dataclass holding `doc_id`, normalized `text`, and rich `metadata`.
- [x] **MS MARCO Dataset Specification:**
  - **Dataset ID:** `microsoft/ms_marco` (updated from legacy `ms_marco` repository ID)
  - **Config:** `v1.1`
  - **Split:** `train`
  - **Selection Method:** Deterministic sequential first-$N$ slice (`train[:sample_size * 2]`). Avoids pseudo-random sampling variance to guarantee reproducibility.
  - **Live Verification Status:** Verified live with `fallback_on_error=False`, confirming download of `microsoft/ms_marco` v1.1 and extraction of records with source `hf_msmarco_v1.1` and ID format `msmarco_{query_id}_{p_idx}`. Seamlessly falls back to local JSONL when network is unavailable.
- [x] **Text Cleaning & Sanitization:** Implemented `clean_passage_text()` featuring NFKC unicode normalization, non-printable control character removal, whitespace collapsing, and newline standardization.
- [x] **Deterministic Content Hashing & Deduplication:** Implemented `compute_content_hash()` using SHA-256 to detect and eliminate duplicate text passages.
- [x] **Quality Filtering & Yield Distinction:** Implemented `clean_and_validate_passages()` enforcing minimum length (`min_length=20`), auto-assigning unique stable IDs, and tracking `char_count` / `word_count`. The actual retained passage count is variable ($\le \text{sample\_size}$) based on duplicate and quality filter yields.
- [x] **Curated Local Fallback Corpus (`data/sample_passages.jsonl`):** 20 high-quality factual passages across machine learning, system architecture, database indexing, networking, and RAG theory for 100% offline reproducibility. Verified and active.

### 2. Chunking Strategies (`rag/chunking.py`)
- [x] **Strict Lineage & Parent ID Preservation:** Defined `Chunk` model preserving parent `doc_id`, `chunk_index`, `strategy`, and inherited metadata. Formats chunk IDs deterministically: `f"{doc_id}_f{idx}"` (fixed) and `f"{doc_id}_s{idx}"` (sentence).
- [x] **Strategy 1 — Fixed-Size Chunking (`FixedSizeChunker`):** Configurable `chunk_size` (default 300) and `chunk_overlap` (default 50) with backwards word-boundary snapping to avoid cutting words in half.
- [x] **Strategy 2 — Sentence-Boundary Chunking (`SentenceBoundaryChunker`):** Alternative structural strategy grouping sentences into target lengths based on grammatical terminals (`.`, `?`, `!`), ensuring complete thoughts are never severed.
- [x] **Edge Case Guardrails:** Validates positive `chunk_size`, non-negative `chunk_overlap`, and enforces `chunk_overlap < chunk_size`. Correctly handles short texts (single chunk) and empty texts (empty list).
- [x] **Corpus Chunking API:** Implemented `chunk_document()` and `chunk_corpus()` for high-throughput batch preprocessing.

### 3. Evaluation Dataset (`data/eval_queries.json`)
- [x] Created 15 labelled test queries categorized by domain, each explicitly mapped to ground-truth relevant passage IDs (`relevant_doc_ids`) in the corpus.
- [x] Validated that all ground-truth passage IDs match real documents in `sample_passages.jsonl` for exact Recall@K scoring in Phase 3.

### 4. Automated Test Results
- **Full Test Suite:** 30 unit tests passing in **0.53s**!
  - `tests/test_ingestion.py` (6 tests): Text cleaning, content hash deduplication, short/empty filtering, local fallback loading, MS MARCO error fallback, evaluation dataset integrity.
  - `tests/test_chunking.py` (8 tests): Fixed-size chunking, sentence-boundary chunking, parameter validation, short/empty passage edge cases, unified corpus chunking.
  - `tests/test_config.py` (4 tests): Masking, defaults, overrides, validation.
  - `tests/test_providers.py` (12 tests): Mock adapters, missing credentials, Ollama `/api/embed`, staging prefix support, honesty checks.

---

## Git & Deployment Status
- **GitHub Repository:** `https://github.com/AnoushkaI/RAGpipeline`
- **Active Branch:** `main`
- **Recent Commits:**
  - `f76dc59` (Phase 1 foundation & provider setup)
  - `71d91ed` (Phase 2 core: MS MARCO ingestion, cleaning, chunking, eval dataset)
  - `9e266a2` (Phase 2 docs & reproducibility cleanup: dataset specs, yield distinction, seed removal)
- **Secrets Protection:** Root `.gitignore` and `yotta-rag-pipeline/.gitignore` actively exclude all `.env` files and credentials.

---

---

## Phase 3 Execution Record: Dense & BM25 Indexing / Retrieval

### 1. In-Memory Dense Vector Indexing (`rag/indexing.py`)
- [x] **Embedding Provider Integration:** Reused the working `SentenceTransformerEmbeddingProvider` (`all-MiniLM-L6-v2`, 384 dimensions) running on CPU.
- [x] **NumPy Vector Matrix (`DenseIndex`):** Stores embeddings in a contiguous 2D float32 array, retaining chunk IDs, parent `doc_id` lineage, chunk text, and metadata.
- [x] **Safe $L_2$ Normalization:** Implemented `safe_l2_normalize()` with an $\epsilon = 10^{-12}$ guardrail to prevent division by zero or NaN creation.
- [x] **Robust Validation:** Validates matrix dimensions, batching limits, and empty chunk lists.

### 2. BM25 Lexical Keyword Indexing (`rag/indexing.py`)
- [x] **`rank-bm25` Integration:** Implemented `BM25Index` wrapping `BM25Okapi` ($k_1=1.5, b=0.75$).
- [x] **Consistent Tokenization:** Implemented `tokenize_text()` extracting lowercase alphanumeric tokens (`\b\w+\b`) applied identically at index and query time.
- [x] **Shared Chunk Corpus:** BM25 is built over the exact same chunks as the dense index to ensure completely fair comparative evaluation.

### 3. Retrieval Engines (`rag/retrieval.py`)
- [x] **Dense Vector Retriever (`DenseRetriever`):** Embeds queries with `SentenceTransformerEmbeddingProvider`, normalizes safely, and computes cosine similarity via vectorized dot products (`np.dot(embeddings, query_vector)`).
- [x] **BM25 Lexical Retriever (`BM25Retriever`):** Tokenizes query, queries `BM25Okapi`, and ranks chunks descending by BM25 score.
- [x] **Standardized Result Model (`RetrievalResult`):** Standard container holding `chunk_id`, `doc_id`, `text`, `score`, `metadata`, and `strategy` provenance.
- [x] **Edge Case Guardrails:** Handles empty/whitespace queries, empty indices, $k \le 0$, zero vectors, and dimension mismatches safely without crashing.

### 4. Fair Comparative Evaluation & Recall@K Benchmark (`rag/evaluation.py`)
- [x] **Strict Parent Lineage Mapping:** Maps retrieved chunk IDs to parent document IDs before computing Recall@K.
- [x] **No Duplicate Chunk Inflation:** Takes the unique set of parent document IDs in top-$K$, guaranteeing that multiple chunks from the same document are **never counted more than once**.
- [x] **15 Labelled Evaluation Queries:** Evaluated across `data/eval_queries.json` at cutoff levels $K \in \{1, 3, 5\}$.
- [x] **Benchmark Runner CLI:** Fully runnable via `python rag/evaluation.py` or `python -m rag.evaluation`.

### 5. Interactive Manual Retrieval Test CLI (`scripts/manual_retrieval_test.py`)
- [x] **Interactive CLI Utility:** Created `scripts/manual_retrieval_test.py` to allow live side-by-side terminal verification of Dense and BM25 retrievers.
- [x] **Execution Command:** `python scripts/manual_retrieval_test.py` (accepts interactive `input()`, empty input exit, or CLI query arguments).
- [x] **CPU Execution:** Uses real `all-MiniLM-L6-v2` embeddings on CPU and BM25Okapi over the 20 local passages (40 chunks).
- [x] **Ranked Output:** Displays top-5 candidates per method with Rank, Score (Cosine/BM25), Chunk ID, Parent Doc ID, and full passage text.

### 6. Actual Benchmark Evaluation Results

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

### 7. Automated Test Suite (50 Tests Passing in ~29s)
- `tests/test_indexing.py` (7 tests): Tokenization, $L_2$ normalization, dense matrix dimensions, empty chunks, BM25 indexing, unified builder.
- `tests/test_retrieval.py` (6 tests): Dense cosine ranking, BM25 ranking, zero/empty queries, dimension mismatch error, facade retrieval.
- `tests/test_evaluation.py` (6 tests): Recall@K math, single hits, duplicate chunk protection, edge cases, eval queries schema, local SentenceTransformer integration test.
- `tests/test_chunking.py` (8 tests): Fixed-size, sentence-boundary, parameters, empty texts.
- `tests/test_ingestion.py` (7 tests): Sanitization, content hashing, deduplication, JSONL loading, error fallback, and `microsoft/ms_marco` repository ID mock verification.
- `tests/test_config.py` (4 tests): Masking, validation, defaults, env overrides.
- `tests/test_providers.py` (12 tests): Mock adapters, missing credentials, Ollama `/api/embed`, staging prefix, honesty checks.

---

## Git & Deployment Status
- **GitHub Repository:** `https://github.com/AnoushkaI/RAGpipeline`
- **Active Branch:** `main`
- **Recent Commits:**
  - `f76dc59` (Phase 1 foundation & provider setup)
  - `71d91ed` (Phase 2 core: MS MARCO ingestion, cleaning, chunking, eval dataset)
  - `9e266a2` (Phase 2 docs & reproducibility cleanup: dataset specs, yield distinction, seed removal)
  - `f2e91f7` (Docs: commit history sync in progress notes)
- **Secrets Protection:** Root `.gitignore` and `yotta-rag-pipeline/.gitignore` actively exclude all `.env` files and credentials.

---

## Next Steps for Phase 4: Relevance Grading, Query Rewriting & Generation (35 min)
1. **Relevance Grader (`rag/grading.py`):**
   - Implement score-threshold based filtering (e.g. cosine score $\ge 0.5$) and LLM-assisted binary grading (`YES` / `NO`).
2. **Query Rewriting:**
   - When retrieved context fails relevance grading, trigger a targeted LLM query reformulation (bounded to at most 1 retry) before attempting a second retrieval pass.
3. **Grounded Answer Generation (`rag/generation.py`):**
   - Prompt the configured LLM (`nvidia/nemotron-3.5-bolt-e4b` or fallback) with strict grounding instructions and explicit bracketed citations (`[doc_id]`).
   - Hallucination safeguard: instruct model to state "Insufficient evidence in retrieved passages" if relevant facts are missing.
