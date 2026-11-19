# PROJECT PROGRESS: Yotta Workstream A — RAG Pipeline

**Project:** Yotta Senior AI Engineer Take-Home • Workstream A: RAG Pipeline  
**Budget:** 180 minutes • 5 Phases  
**Current Phase:** Phase 2 Complete (Awaiting User Approval for Phase 3)  
**Last Updated:** Phase 2 Completion  

---

## Phase Status Summary

| Phase | Description | Allocated | Status | Gate Check / Deliverables |
|---|---|---|---|---|
| **Phase 1** | **Project Foundation & Provider Setup** | **25 min** | **COMPLETED & HARDENED** | Modular structure, provider abstraction, health checks, config & tests |
| **Phase 2** | **MS MARCO Ingestion, Cleaning & Chunking** | **40 min** | **COMPLETED** | Reproducible subset, local fallback JSONL, 2 chunking strategies, eval dataset |
| **Phase 3** | Dense & BM25 Indexing / Retrieval | 45 min | Pending Approval | In-memory NumPy cosine similarity, BM25 index, Recall@K benchmark |
| **Phase 4** | Relevance Grading, Rewrite & Generation | 35 min | Pending Phase 3 | Threshold/LLM grader, 1 bounded retry, grounded citation prompt |
| **Phase 5** | Streamlit UI, Tests, Evaluation & Docs | 35 min | Pending Phase 4 | Interactive UI, failure analysis, latency profile, full report |

---

## Phase 2 Execution Record: Ingestion, Cleaning & Chunking

### 1. Ingestion & Preprocessing Architecture (`rag/ingestion.py`)
- [x] **Passage Data Structure:** Implemented `Passage` dataclass holding `doc_id`, normalized `text`, and rich `metadata`.
- [x] **Text Cleaning & Sanitization:** Implemented `clean_passage_text()` featuring NFKC unicode normalization, non-printable control character removal, whitespace collapsing, and newline standardization.
- [x] **Deterministic Content Hashing & Deduplication:** Implemented `compute_content_hash()` using SHA-256 to detect and eliminate duplicate text passages.
- [x] **Quality Filtering & Schema Validation:** Implemented `clean_and_validate_passages()` enforcing minimum length threshold (`min_length=20`), auto-assigning unique stable IDs when missing, and recording character/word counts into metadata.
- [x] **Curated Local Fallback Corpus (`data/sample_passages.jsonl`):** Created 20 high-quality factual passages across machine learning, system architecture, database indexing, networking, and RAG theory to guarantee 100% offline reproducibility.
- [x] **MS MARCO Subset Ingestion (`load_msmarco_subset`):** Added deterministic subset extraction from Hugging Face Datasets with automatic resilient fallback to local JSONL whenever remote access is constrained.

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
- **Initial Sync Commit:** `f76dc59` (Phase 1 foundation & provider setup)
- **Secrets Protection:** Root `.gitignore` and `yotta-rag-pipeline/.gitignore` actively exclude all `.env` files and credentials.

---

## Next Steps for Phase 3: Dense & BM25 Indexing / Retrieval (45 min)
1. **Dense Vector Indexing:** Generate chunk embeddings using the working Sentence-Transformers model (`all-MiniLM-L6-v2`) and store vectors in an in-memory NumPy matrix.
2. **Dense Cosine Retrieval:** Embed incoming user queries and rank top-$k$ candidate chunks using vectorized cosine similarity.
3. **BM25 Lexical Indexing:** Tokenize chunks and construct an inverted BM25 index using `rank-bm25`.
4. **Lexical Retrieval:** Retrieve and rank top-$k$ chunks for the same query using BM25.
5. **Retrieval Comparison & Recall@K:** Run the 15 labelled evaluation queries through both retrievers and compute Recall@K comparison benchmarks.
