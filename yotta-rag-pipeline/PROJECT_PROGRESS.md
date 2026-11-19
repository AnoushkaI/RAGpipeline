# PROJECT PROGRESS: Yotta Workstream A — RAG Pipeline

**Project:** Yotta Senior AI Engineer Take-Home • Workstream A: RAG Pipeline  
**Budget:** 180 minutes • 5 Phases  
**Current Phase:** Phase 4 Complete (Ready for Phase 5: Streamlit UI & Evaluation Dashboard)  
**Last Updated:** Phase 4 Completion  

---

## Phase Status Summary

| Phase | Description | Allocated | Status | Gate Check / Deliverables |
|---|---|---|---|---|
| **Phase 1** | **Project Foundation & Provider Setup** | **25 min** | **COMPLETED & HARDENED** | Modular structure, provider abstraction, health checks, config & tests |
| **Phase 2** | **MS MARCO Ingestion, Cleaning & Chunking** | **40 min** | **COMPLETED** | Reproducible subset, local fallback JSONL, 2 chunking strategies, eval dataset |
| **Phase 3** | **Dense & BM25 Indexing / Retrieval** | **45 min** | **COMPLETED** | In-memory NumPy cosine similarity, BM25 index, fair Recall@K benchmark |
| **Phase 4** | **Relevance Grading, Rewrite & Generation** | **35 min** | **COMPLETED** | Threshold/LLM grader, 1 bounded retry, hybrid RRF, grounded citation prompt |
| **Phase 5** | **Streamlit UI, Tests, Evaluation & Docs** | **35 min** | Pending Approval | Interactive UI, failure analysis, latency profile, full report |

---

## Phase 4 Execution Record: End-to-End Pipeline, Grading, Rewrite & Generation

### 1. Relevance Grading Architecture (`rag/grading.py`)
- [x] **Configurable & Visible Grading Methods:** Implemented `RelevanceGrader` supporting two operational modes:
  - **`threshold` Method:** Fast numerical evaluation against configured threshold (`relevance_threshold`, default 0.5) over Dense cosine similarity, BM25 scores, and normalized Hybrid scores.
  - **`llm` Method:** Zero-shot/few-shot semantic classification prompting the generation model to return a binary `YES` or `NO` verdict evaluating whether the passage contains actionable evidence.
- [x] **Detailed Diagnostics:** Emits structured `GradingDecision` objects capturing `chunk_id`, parent `doc_id`, raw score, relevance status, and explanation reasoning.
- [x] **Consolidated Result Container:** Implemented `GradingResult` separating `passed_chunks` from `rejected_chunks`, recording `is_sufficient: bool`, and providing serializable `.to_dict()`.
- [x] **Graceful Fallback:** If the LLM grader encounters a connection error or timeout, it automatically falls back to threshold evaluation and logs a warning without crashing the pipeline.

### 2. Strictly Grounded Generation (`rag/generation.py`)
- [x] **Strict Anti-Hallucination Grounding Prompt:** Enforces reliance *only* on provided context passages. Forbids answering from pretrained knowledge or extrapolating beyond explicit facts.
- [x] **Single Consolidated Synthesis:** Explicitly directs the model to synthesize a single cohesive, well-written answer combining evidence from multiple passages, avoiding fragmented or disconnected replies.
- [x] **Mandatory Source Attribution:** Requires bracketed parent document citations (e.g. `[doc_0001]`) following each factual claim.
- [x] **Strict Citation Verification:** Implemented regex extraction (`extract_citations()`) validated against the retrieved chunk corpus. Any hallucinated citation IDs that do not exist in the active context are stripped out.
- [x] **Explicit Abstention:** When context is empty or evidence is missing, the generator immediately emits a standardized abstention message:
  > *"I cannot answer this question based on the provided documents as the necessary information is not present in the retrieved context."*

### 3. Multi-Strategy Retrieval & Hybrid Fusion (`rag/retrieval.py`)
- [x] **Dense Vector Retrieval:** Cosine similarity via NumPy dot-products over $L_2$-normalized matrices.
- [x] **BM25 Lexical Keyword Retrieval:** `rank-bm25` (BM25Okapi) scoring over identical tokenized chunks.
- [x] **Hybrid Retrieval (`HybridRetriever`):** Combines dense vector semantic similarity with BM25 keyword rankings via **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
  Normalized to $[0.0, 1.0]$ for consistent threshold grading across all retrieval strategies.

### 4. Orchestration Pipeline & Bounded Retry (`rag/pipeline.py`)
- [x] **End-to-End Coordination:** Links Ingestion $\to$ Chunking $\to$ Indexing $\to$ Retrieval $\to$ Relevance Grading $\to$ Bounded Retry $\to$ Grounded Generation.
- [x] **Strictly Bounded Query Rewrite (Max 1 Retry):** If initial retrieval yields insufficient context, the pipeline reformulates the user's query using the LLM (`rewrite_query()`), executes a second retrieval pass, and re-grades candidates. Bounded strictly to at most 1 retry to prevent infinite loops.
- [x] **Defensive Error Handling:**
  - Empty or whitespace queries return an immediate validation notice with zero latency overhead.
  - Unindexed corpus state returns an informative warning without raising runtime exceptions.
  - Provider execution timeouts and HTTP errors during generation are caught and reported cleanly as structured diagnostic failures.
- [x] **Structured Output (`RAGResponse`):** Returns `query`, `answer`, `source_passages` (`passed_chunks`), `cited_doc_ids`, `retrieved_chunks`, `strategy`, `has_sufficient_context`, `retry_flag` (`retry_attempted`), `rewritten_query`, `timing` (`latency_ms`), `grading_method`, and `grading_result` (`grading_details`). Supports both attribute and dict-like indexing access.
- [x] **Audit & Verification Checks:**
  - Insufficient context triggers exactly 1 rewrite and 1 retry, strictly bounded.
  - Only accepted passages and parent source IDs reach generation; rejected chunks are excluded from prompts.
  - Citation IDs are validated against active context, discarding hallucinated IDs.
  - Generation abstains immediately when evidence remains insufficient after retry.
  - **Relevance Grading Calibration (BM25 & Hybrid):** Fixed raw BM25 score comparison bug where raw scores (e.g. 6.32) exceeded cosine threshold (0.5). Implemented `compute_bm25_relevance()` mapping lexical evidence to $[0.0, 1.0]$ via content-term coverage and score scaling. Hybrid grading now validates underlying dense and lexical signals to reject irrelevant chunks ranked top by pure RRF.
- [x] **Interactive & Automated End-to-End CLI (`scripts/manual_rag_test.py`):** Created full pipeline manual test runner demonstrating both unsupported ("Yotta market share") and supported ("Supervised vs unsupervised learning") queries with full diagnostic traces across Dense, BM25, and Hybrid.

### 5. Automated Test Suite (63 Tests Passing)
- **`tests/test_pipeline.py` (13 tests):**
  - End-to-end question answering with verified citations.
  - Irrelevant query retry trigger and graceful abstention.
  - Dense high-threshold retry trigger.
  - Empty/whitespace query validation.
  - Unindexed pipeline handling.
  - Multi-strategy retrieval (dense, bm25, hybrid).
  - Relevance grading modes (threshold vs LLM).
  - Provider failure / timeout resilience.
  - Citation hallucination filtering.
  - Empty context generator abstention.
  - Bounded retry verification (strictly 1 rewrite & retry, never an unbounded loop).
  - Context isolation verification (only accepted passages and source IDs reach generation).
  - **Regression Test:** `test_unsupported_question_rejection_across_strategies` verifying rejection across Dense, BM25, and Hybrid.
- **Full Suite Status:** 63 passed out of 63 tests across the entire repository.

---

## Phase 3 Benchmark Summary (Retained for Reference)

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

---

## Git & Deployment Status
- **GitHub Repository:** `https://github.com/AnoushkaI/RAGpipeline`
- **Active Branch:** `main`
- **Recent Commits:**
  - `f76dc59` (Phase 1 foundation & provider setup)
  - `71d91ed` (Phase 2 core: MS MARCO ingestion, cleaning, chunking, eval dataset)
  - `9e266a2` (Phase 2 docs & reproducibility cleanup)
  - `f2e91f7` (Docs: commit history sync in progress notes)
  - `1e038ac` (Phase 3: Dense and BM25 retrieval, evaluation benchmark, and manual test CLI)
- **User Instruction for Git Push:** For Phase 4 and future changes, user will push via terminal `git push origin main`.
- **Secrets Protection:** Root `.gitignore` and `yotta-rag-pipeline/.gitignore` actively exclude all `.env` files and credentials.

---

## Next Steps for Phase 5: Streamlit UI, Full Evaluation & Final Documentation (35 min)
1. **Interactive Streamlit UI (`app.py`):**
   - Question input, strategy selector (`dense`, `bm25`, `hybrid`), top-k slider, and relevance threshold slider.
   - Display grounded consolidated answer, parent document citations, and confidence/grounding badge.
   - Expandable inspection panels: retrieved chunks, relevance grading verdicts, rewritten query (if retry was triggered), and latency breakdown.
2. **Evaluation Framework & Faithfulness Scoring (`rag/evaluation.py`):**
   - Implement claim-by-claim answer faithfulness / groundedness evaluation rubric.
   - Generate evaluation summary artifacts (`results/evaluation_results.csv`).
3. **Failure Analysis & Technical Trade-offs Report:**
   - Document failure modes, latency profiles across CPU vs GPU, and architectural trade-offs.
