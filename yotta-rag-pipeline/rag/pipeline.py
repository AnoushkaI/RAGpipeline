"""
Orchestration Pipeline Module for Yotta Workstream A RAG Pipeline.

Phase 4 Implementation:
- Connects ingestion -> chunking -> indexing -> retrieval -> grading -> retry -> generation.
- Supports configurable retrieval strategies: 'dense', 'bm25', and 'hybrid' (Reciprocal Rank Fusion).
- Enforces relevance grading using configurable threshold or LLM classifier.
- Implements single bounded retry (max 1 query rewrite) when initial retrieval yields insufficient context.
- Generates a single consolidated answer grounded strictly in retrieved evidence, with source document citations.
- Gracefully handles empty queries, empty indices, provider connection/execution errors, and zero evidence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from config import AppConfig, load_config
from rag.chunking import Chunk, FixedSizeChunker, SentenceBoundaryChunker, chunk_corpus
from rag.generation import GroundedGenerator
from rag.grading import RelevanceGrader
from rag.indexing import BM25Index, DenseIndex, build_indices
from rag.ingestion import Passage, load_corpus
from rag.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ModelProvider,
    ProviderError,
    get_embedding_provider,
    get_llm_provider,
    get_model_provider,
)
from rag.retrieval import RetrievalResult, retrieve_top_k

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# RAG Response Data Structure
# ------------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """
    Structured outcome of an end-to-end RAG pipeline run.
    
    Attributes:
        query: Original user query.
        answer: Final consolidated grounded answer or abstention notice.
        cited_doc_ids: Verified parent document IDs cited in the answer.
        retrieved_chunks: Chunks retrieved in the final retrieval stage.
        passed_chunks: Chunks that successfully passed relevance grading.
        strategy: Retrieval strategy used ('dense', 'bm25', or 'hybrid').
        has_sufficient_context: True if sufficient grounded context was available.
        retry_attempted: True if an insufficient-context retry was triggered.
        rewritten_query: Alternative query used during retry (if any).
        latency_ms: Total pipeline execution latency in milliseconds.
        grading_details: Metadata and per-chunk verdicts from relevance grading.
    """
    query: str
    answer: str
    cited_doc_ids: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    passed_chunks: list[RetrievalResult] = field(default_factory=list)
    strategy: str = "dense"
    has_sufficient_context: bool = False
    retry_attempted: bool = False
    rewritten_query: Optional[str] = None
    latency_ms: float = 0.0
    grading_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert response to serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "cited_doc_ids": self.cited_doc_ids,
            "source_passages": [c.to_dict() for c in self.passed_chunks],
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "passed_chunks": [c.to_dict() for c in self.passed_chunks],
            "strategy": self.strategy,
            "has_sufficient_context": self.has_sufficient_context,
            "retry_attempted": self.retry_attempted,
            "retry_flag": self.retry_attempted,
            "rewritten_query": self.rewritten_query,
            "latency_ms": round(self.latency_ms, 2),
            "timing": round(self.latency_ms, 2),
            "grading_method": self.grading_method,
            "grading_result": self.grading_details,
            "grading_details": self.grading_details,
        }

    @property
    def source_passages(self) -> list[RetrievalResult]:
        """Expose accepted source passages used for grounded context generation."""
        return self.passed_chunks

    @property
    def grading_result(self) -> dict[str, Any]:
        """Expose grading result metadata and decisions."""
        return self.grading_details

    @property
    def grading_method(self) -> str:
        """Expose chosen grading method ('threshold' or 'llm')."""
        return str(self.grading_details.get("method", "threshold"))

    @property
    def retry_flag(self) -> bool:
        """Alias for retry_attempted."""
        return self.retry_attempted

    @property
    def timing(self) -> float:
        """Alias for latency_ms."""
        return self.latency_ms

    def __getitem__(self, item: str) -> Any:
        """Allow dict-like item access for backward-compatibility."""
        return getattr(self, item)


# ------------------------------------------------------------------------------
# End-to-End RAG Pipeline Orchestrator
# ------------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the complete lifecycle of a Retrieval-Augmented Generation query:
    Retrieval -> Grading -> Bounded Retry (if insufficient) -> Grounded Generation.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        dense_index: Optional[DenseIndex] = None,
        bm25_index: Optional[BM25Index] = None,
        grader: Optional[RelevanceGrader] = None,
        generator: Optional[GroundedGenerator] = None,
        passages: Optional[list[Passage]] = None,
        chunks: Optional[list[Chunk]] = None,
        chunking_strategy: Literal["fixed", "sentence"] = "fixed",
    ):
        self.config = config or load_config()

        # Initialize providers
        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = get_llm_provider(self.config.llm)

        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            self.embedding_provider = get_embedding_provider(self.config.embedding)

        # Grader & Generator
        self.grader = grader or RelevanceGrader(
            method="threshold",
            threshold=self.config.rag.relevance_threshold,
            llm_provider=self.llm_provider,
        )
        self.generator = generator or GroundedGenerator(llm_provider=self.llm_provider)

        # Indices
        self.dense_index = dense_index
        self.bm25_index = bm25_index

        # Auto-index if chunks or passages supplied at initialization
        if chunks:
            self.index_chunks(chunks)
        elif passages:
            self.index_passages(passages, strategy=chunking_strategy)

    def index_passages(
        self,
        passages: list[Passage],
        strategy: Literal["fixed", "sentence"] = "fixed",
    ) -> tuple[DenseIndex, BM25Index]:
        """
        Chunks passages using the selected strategy and builds Dense and BM25 indices.
        """
        chunks = chunk_corpus(
            passages=passages,
            strategy=strategy,
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
        )
        return self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> tuple[DenseIndex, BM25Index]:
        """
        Builds in-memory Dense and BM25 indices over the supplied chunks.
        """
        d_idx, b_idx = build_indices(chunks=chunks, embedding_provider=self.embedding_provider)
        self.dense_index = d_idx
        self.bm25_index = b_idx
        logger.info(f"Indexed {len(chunks)} chunks in Dense and BM25 indices.")
        return d_idx, b_idx

    def rewrite_query(self, query: str) -> str:
        """
        Reformulates the user's query when the initial retrieval yields insufficient context.
        Uses the LLM provider if available, or applies robust keyword expansion fallback.
        """
        prompt = (
            "You are an expert search query reformulator.\n"
            f"The initial search for the query '{query}' yielded no relevant passages.\n"
            "Generate an improved, concise alternative search query using synonyms, "
            "related terminology, or expanded keywords to find relevant evidence.\n"
            "Output ONLY the new query string, without quotes or additional commentary."
        )
        try:
            raw_rewritten = self.llm_provider.generate(prompt)
            clean_rewritten = raw_rewritten.strip().strip('"').strip("'")
            if clean_rewritten and len(clean_rewritten) > 2:
                return clean_rewritten
        except Exception as e:
            logger.warning(f"Query rewrite LLM call failed: {e}. Using rule-based fallback.")

        # Fallback query variation
        words = query.strip().split()
        if len(words) > 1:
            return " ".join(words[::-1])  # Simple permutation
        return query

    def run(
        self,
        query: str,
        strategy: Optional[Literal["dense", "bm25", "hybrid"]] = None,
        top_k: Optional[int] = None,
        grading_method: Optional[Literal["threshold", "llm"]] = None,
        threshold: Optional[float] = None,
    ) -> RAGResponse:
        """
        Execute end-to-end question answering pipeline for a query.
        
        Args:
            query: User's question or search query.
            strategy: 'dense', 'bm25', or 'hybrid' (defaults to 'dense').
            top_k: Number of chunks to retrieve (defaults to config.rag.top_k).
            grading_method: Optional override for grading method ('threshold' or 'llm').
            threshold: Optional override for relevance threshold.
            
        Returns:
            RAGResponse object containing answer, citations, timing, and diagnostics.
        """
        start_time = time.perf_counter()
        active_strategy = (strategy or "dense").lower().strip()
        active_k = top_k if (top_k is not None and top_k > 0) else self.config.rag.top_k

        # 1. Guard against empty/whitespace query
        clean_query = query.strip() if query else ""
        if not clean_query:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                query=query,
                answer="Please provide a valid, non-empty query.",
                cited_doc_ids=[],
                retrieved_chunks=[],
                passed_chunks=[],
                strategy=active_strategy,
                has_sufficient_context=False,
                retry_attempted=False,
                rewritten_query=None,
                latency_ms=elapsed_ms,
                grading_details={"reason": "Empty or whitespace query provided."},
            )

        # 2. Check index initialization
        has_dense = self.dense_index is not None and not self.dense_index.is_empty
        has_bm25 = self.bm25_index is not None and not self.bm25_index.is_empty

        if (active_strategy == "dense" and not has_dense) or \
           (active_strategy == "bm25" and not has_bm25) or \
           (active_strategy == "hybrid" and not (has_dense and has_bm25)):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                query=clean_query,
                answer="No documents have been indexed yet in the RAG pipeline.",
                cited_doc_ids=[],
                retrieved_chunks=[],
                passed_chunks=[],
                strategy=active_strategy,
                has_sufficient_context=False,
                retry_attempted=False,
                rewritten_query=None,
                latency_ms=elapsed_ms,
                grading_details={"reason": "Target index is empty or uninitialized."},
            )

        # Helper for retrieval execution
        def _do_retrieve(q: str) -> list[RetrievalResult]:
            if active_strategy == "dense":
                assert self.dense_index is not None
                return retrieve_top_k(
                    query=q,
                    index=self.dense_index,
                    strategy="dense",
                    embedding_provider=self.embedding_provider,
                    top_k=active_k,
                )
            elif active_strategy == "bm25":
                assert self.bm25_index is not None
                return retrieve_top_k(
                    query=q,
                    index=self.bm25_index,
                    strategy="bm25",
                    top_k=active_k,
                )
            elif active_strategy == "hybrid":
                assert self.dense_index is not None and self.bm25_index is not None
                return retrieve_top_k(
                    query=q,
                    index=(self.dense_index, self.bm25_index),
                    strategy="hybrid",
                    embedding_provider=self.embedding_provider,
                    top_k=active_k,
                )
            else:
                raise ValueError(f"Unknown retrieval strategy '{active_strategy}'")

        # 3. Initial Retrieval
        try:
            candidates = _do_retrieve(clean_query)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                query=clean_query,
                answer=f"Retrieval failed due to error: {str(e)}",
                cited_doc_ids=[],
                retrieved_chunks=[],
                passed_chunks=[],
                strategy=active_strategy,
                has_sufficient_context=False,
                retry_attempted=False,
                rewritten_query=None,
                latency_ms=elapsed_ms,
                grading_details={"error": str(e)},
            )

        # 4. Relevance Grading
        active_threshold = float(threshold) if threshold is not None else self.config.rag.relevance_threshold
        active_method = (grading_method or self.grader.method).lower().strip()
        if active_method != self.grader.method or active_threshold != self.grader.threshold:
            active_grader = RelevanceGrader(
                method=active_method,
                threshold=active_threshold,
                llm_provider=self.llm_provider,
            )
        else:
            active_grader = self.grader

        grading_result = active_grader.grade(query=clean_query, candidates=candidates)

        retry_attempted = False
        rewritten_query = None

        # 5. Insufficient Context & Single Bounded Retry Check
        # Bounded to exactly 1 retry max if initial context is insufficient
        if not grading_result.is_sufficient and self.config.rag.max_retries >= 1:
            logger.info("Initial retrieval yielded insufficient context. Triggering bounded query retry...")
            retry_attempted = True
            rewritten_query = self.rewrite_query(clean_query)

            # Retrieve with rewritten query
            try:
                retry_candidates = _do_retrieve(rewritten_query)
            except Exception as e:
                logger.warning(f"Retry retrieval failed: {e}")
                retry_candidates = []

            # Grade retry candidates
            retry_grading = active_grader.grade(query=rewritten_query, candidates=retry_candidates)

            if retry_grading.is_sufficient:
                logger.info("Retry retrieval succeeded in finding relevant evidence.")
                candidates = retry_candidates
                grading_result = retry_grading
            else:
                logger.info("Retry retrieval also yielded insufficient context. Abstaining.")
                # We retain candidates for diagnostic reporting
                if retry_candidates:
                    candidates = retry_candidates

        # 6. Final Decision: Grounded Generation vs Abstention
        if not grading_result.is_sufficient:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                query=clean_query,
                answer=(
                    "I cannot answer this question based on the provided documents "
                    "as insufficient relevant context was found."
                ),
                cited_doc_ids=[],
                retrieved_chunks=candidates,
                passed_chunks=[],
                strategy=active_strategy,
                has_sufficient_context=False,
                retry_attempted=retry_attempted,
                rewritten_query=rewritten_query,
                latency_ms=elapsed_ms,
                grading_details=grading_result.to_dict(),
            )

        # 7. Grounded Answer Generation
        gen_result = self.generator.generate(
            query=clean_query,
            context_chunks=grading_result.passed_chunks,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            query=clean_query,
            answer=gen_result.answer,
            cited_doc_ids=gen_result.cited_doc_ids,
            retrieved_chunks=candidates,
            passed_chunks=grading_result.passed_chunks,
            strategy=active_strategy,
            has_sufficient_context=gen_result.has_sufficient_context,
            retry_attempted=retry_attempted,
            rewritten_query=rewritten_query,
            latency_ms=elapsed_ms,
            grading_details=grading_result.to_dict(),
        )


# ------------------------------------------------------------------------------
# Convenience Factory
# ------------------------------------------------------------------------------

def create_pipeline(
    config: Optional[AppConfig] = None,
    load_local_data: bool = True,
    sample_size: int = 20,
) -> RAGPipeline:
    """
    Creates and initializes a ready-to-query RAGPipeline.
    If load_local_data is True, loads and indexes the curated local corpus.
    """
    cfg = config or load_config()
    pipeline = RAGPipeline(config=cfg)

    if load_local_data:
        passages = load_corpus(use_hf=False, sample_size=sample_size)
        pipeline.index_passages(passages=passages, strategy="fixed")

    return pipeline
