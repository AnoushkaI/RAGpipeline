# PROJECT PROGRESS: Yotta Workstream A — RAG Pipeline

**Project:** Yotta Senior AI Engineer Take-Home • Workstream A: RAG Pipeline  
**Budget:** 180 minutes • 5 Phases  
**Current Phase:** Phase 1 Complete & Provider Layer Hardened (Awaiting User Approval for Phase 2)  
**Last Updated:** Phase 1 Hardening & Provider Layer Review  

---

## Phase Status Summary

| Phase | Description | Allocated | Status | Gate Check / Deliverables |
|---|---|---|---|---|
| **Phase 1** | **Project Foundation & Provider Setup** | **25 min** | **COMPLETED & HARDENED** | Modular structure, provider abstraction, health checks, config & tests |
| **Phase 2** | MS MARCO Ingestion, Cleaning & Chunking | 40 min | Pending Approval | Reproducible subset, local fallback, 2 chunking strategies, eval queries |
| **Phase 3** | Dense & BM25 Indexing / Retrieval | 45 min | Pending Phase 2 | In-memory NumPy cosine similarity, BM25 index, Recall@K benchmark |
| **Phase 4** | Relevance Grading, Rewrite & Generation | 35 min | Pending Phase 3 | Threshold/LLM grader, 1 bounded retry, grounded citation prompt |
| **Phase 5** | Streamlit UI, Tests, Evaluation & Docs | 35 min | Pending Phase 4 | Interactive UI, failure analysis, latency profile, full report |

---

## Phase 1 Execution & Provider Hardening Record

### 1. Provider Layer Review & Enhancements (`rag/providers.py`)
- [x] **Preserved Architecture:** Retained `BaseLLMProvider`, `BaseEmbeddingProvider`, and `ModelProvider` decoupled abstractions, factory functions, and existing provider signatures.
- [x] **NVIDIA NIM Generation Live Verification:** Tested and verified live inference using the OpenAI-compatible chat completions API with `nvidia/nemotron-3.5-bolt-e4b`.
- [x] **Standalone Gateway Staging Prefix Support:** Implemented `_candidate_models()` supporting both standard (`nvidia/nemotron-3.5-bolt-e4b`) and staging gateway prefixes (`stg/nvidia/nemotron-3.5-bolt-e4b`), with automatic 404 fallback routing.
- [x] **Live Generation Health Checks:** Replaced simple `models.list()` calls with actual lightweight model generation probes (`max_tokens=10`), verifying end-to-end inference readiness.
- [x] **Strict Secret Sanitization:** Masked all API keys (`nvapi-...`, `sk-...`, `Bearer ...`) from exception messages and logs to guarantee zero credential leakage.
- [x] **Generation vs. Embedding Separation:** Enforced clear separation preventing Nemotron Bolt generation models from being mistakenly used as embedding models, keeping `sentence_transformers` (`all-MiniLM-L6-v2`) as the independent default.
- [x] **Modern Ollama Embeddings (`/api/embed`):** Updated `OllamaEmbeddingProvider` to use batched input against `/api/embed`, correctly parsing `embeddings`, with graceful fallback to `/api/embeddings`.
- [x] **Ollama Installed Model Verification:** Upgraded Ollama health checks to inspect `/api/tags` and verify that the target model is installed, returning `FAILED` with actionable `ollama pull <model>` instructions when missing.
- [x] **Honest Smoke Test Reporting:** Updated `run_provider_smoke_test()` to return failure (`exit code 1`) on `FAILED`, `MISSING_CREDENTIALS`, or `UNTESTED` states, never misrepresenting incomplete setups as successful.

### 2. Provider Readiness Matrix

| Provider | Role | Model Identifier | Live Health Check Status | Notes |
|---|---|---|---|---|
| **NVIDIA NIM** | LLM Generation | `nvidia/nemotron-3.5-bolt-e4b` | **`OK` (VERIFIED LIVE)** | Live response verified via chat completions; supports `stg/` prefix |
| **Sentence-Transformers** | Dense Embeddings | `all-MiniLM-L6-v2` | **`OK` (VERIFIED LOCAL)** | Verified locally on CPU; snapshot cache resolved |
| **Mock Providers** | LLM & Embeddings | `mock-model` / `mock-embed` | **`OK` (VERIFIED)** | Fast deterministic test adapters for offline CI |
| **OpenAI** | LLM & Embeddings | `gpt-4o-mini` / `text-embedding-3-small` | **`CONFIGURED`** | Tested graceful `MISSING_CREDENTIALS` error handling |
| **Llama (Ollama)** | LLM & Embeddings | `llama3.2` / `nomic-embed-text` | **`CONFIGURED`** | Verified batched `/api/embed` & missing-model reporting |

### 3. Automated Test Results
- **Provider & Config Test Suite:** 16 tests passing in 0.44s (`tests/test_providers.py`, `tests/test_config.py`).
- **Targeted Smoke Tests:**
  - Live NVIDIA NIM generation: `[OK] NVIDIA NIM (generation - nvidia/nemotron-3.5-bolt-e4b): Verified live generation via chat completions API.`
  - Staging prefix candidate resolution: Automatically resolved `stg/` $\leftrightarrow$ bare model name on 404.
  - Honest missing model / unreachable Ollama reporting: Reported `[FAILED]` with `exit code 1`.

### 4. Configuration Changes in `.env`
To activate NVIDIA NIM Nemotron generation as the default LLM, update the following in `.env` (no secrets are displayed):
```env
LLM_PROVIDER=nvidia_nim
NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Next Steps for Phase 2 (Pending User Approval)
1. **MS MARCO Dataset:** Load a small reproducible subset using Hugging Face `datasets` with deterministic slicing.
2. **Local Fallback:** Create `data/sample_passages.jsonl` so the pipeline remains 100% runnable offline.
3. **Passage Cleaning:** Normalization, deduplication, removal of empty passages, stable passage IDs (`doc_id`).
4. **Chunking Strategies:** Implement fixed-size chunking with overlap and sentence-boundary chunking, maintaining parent passage ID lineage (`doc_id` $\rightarrow$ `chunk_id`).
5. **Evaluation Set:** Create `data/eval_queries.json` with labelled questions and ground-truth passage IDs.
