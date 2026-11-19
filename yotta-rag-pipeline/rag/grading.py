"""
Relevance Grading Module for Yotta Workstream A RAG Pipeline.

Phase 4 Implementation:
- Evaluates retrieved candidate chunks against the user query.
- Supports two visible, configurable grading methods:
  1. 'threshold': Fast numerical threshold evaluation based on similarity/relevance scores.
  2. 'llm': Few-shot / zero-shot LLM binary classification (YES/NO) evaluating semantic relevance.
- Determines whether sufficient relevant context exists to ground an answer,
  or if query rewriting and retry should be triggered.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from rag.indexing import tokenize_text
from rag.providers import BaseLLMProvider, ProviderError
from rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# Standard English stopwords to filter out meaningless token matches during lexical evaluation
ENGLISH_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
}


def compute_bm25_relevance(query: str, cand_text: str, raw_score: float) -> tuple[float, float, str]:
    """
    Computes a normalized BM25 relevance score in [0.0, 1.0] based on query term coverage
    and BM25 keyword score, preventing raw unbounded BM25 scores from being compared
    directly with cosine similarity thresholds.
    
    Returns:
        (normalized_score, coverage_ratio, explanation_reason)
    """
    raw_q_tokens = tokenize_text(query)
    q_content_tokens = [t for t in raw_q_tokens if t not in ENGLISH_STOPWORDS]
    if not q_content_tokens:
        q_content_tokens = raw_q_tokens

    if not q_content_tokens or raw_score <= 0.0:
        return 0.0, 0.0, f"BM25 non-positive score ({raw_score:.2f}) or empty query tokens"

    doc_tokens = set(tokenize_text(cand_text))
    matched_tokens = set(q_content_tokens) & doc_tokens
    coverage = len(matched_tokens) / len(set(q_content_tokens))

    # Score saturation factor: requires positive BM25 signal (saturates at 3.0)
    score_factor = min(1.0, max(0.0, raw_score / 3.0))
    norm_score = coverage * score_factor

    reason = (
        f"BM25 normalized {norm_score:.4f} (coverage={coverage:.2f} [{len(matched_tokens)}/{len(set(q_content_tokens))} terms], "
        f"raw_score={raw_score:.2f})"
    )
    return norm_score, coverage, reason


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass
class GradingDecision:
    """Individual chunk relevance grading decision."""
    chunk_id: str
    doc_id: str
    score: float
    is_relevant: bool
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GradingResult:
    """
    Consolidated relevance grading output.
    
    Attributes:
        passed_chunks: List of candidate chunks that passed the relevance filter.
        rejected_chunks: List of candidate chunks deemed irrelevant.
        is_sufficient: True if at least one chunk passed grading with usable evidence.
        method: Chosen grading method ('threshold' or 'llm').
        threshold: Score threshold used if threshold method was applied.
        decisions: Detailed per-chunk grading decisions.
    """
    passed_chunks: list[RetrievalResult] = field(default_factory=list)
    rejected_chunks: list[RetrievalResult] = field(default_factory=list)
    is_sufficient: bool = False
    method: str = "threshold"
    threshold: float = 0.5
    decisions: list[GradingDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed_chunks": [c.to_dict() for c in self.passed_chunks],
            "rejected_chunks": [c.to_dict() for c in self.rejected_chunks],
            "is_sufficient": self.is_sufficient,
            "method": self.method,
            "threshold": self.threshold,
            "decisions": [d.to_dict() for d in self.decisions],
        }


# ------------------------------------------------------------------------------
# Relevance Grader
# ------------------------------------------------------------------------------

class RelevanceGrader:
    """
    Filters retrieved candidate chunks to ensure only relevant evidence is passed
    to the generation step.
    """

    def __init__(
        self,
        method: str = "threshold",
        threshold: float = 0.5,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.method = method.lower().strip()
        if self.method not in {"threshold", "llm"}:
            raise ValueError(f"Unsupported grading method '{method}'. Choose 'threshold' or 'llm'.")
        self.threshold = float(threshold)
        self.llm_provider = llm_provider

        if self.method == "llm" and self.llm_provider is None:
            logger.warning(
                "LLM grading method selected but no LLM provider supplied. "
                "Falling back to threshold grading."
            )
            self.method = "threshold"

    def grade(self, query: str, candidates: list[RetrievalResult]) -> GradingResult:
        """
        Evaluate candidate chunks and partition into passed and rejected.
        
        Args:
            query: User's original or rewritten question.
            candidates: List of RetrievalResult objects from retriever.
            
        Returns:
            GradingResult summarizing relevance status and passed chunks.
        """
        if not candidates:
            return GradingResult(
                passed_chunks=[],
                rejected_chunks=[],
                is_sufficient=False,
                method=self.method,
                threshold=self.threshold,
                decisions=[],
            )

        if self.method == "llm" and self.llm_provider is not None:
            return self._grade_with_llm(query, candidates)
        else:
            return self._grade_with_threshold(query, candidates)

    def _grade_with_threshold(self, query: str, candidates: list[RetrievalResult]) -> GradingResult:
        passed: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []
        decisions: list[GradingDecision] = []

        # Find max score in candidates for relative scaling when appropriate
        max_score = max((c.score for c in candidates), default=0.0)

        for cand in candidates:
            strat = cand.strategy.lower()
            is_rel = False
            reason = ""

            if strat == "dense":
                # Cosine similarity in [-1.0, 1.0]
                is_rel = cand.score >= self.threshold
                status_str = ">=" if is_rel else "<"
                reason = f"Dense cosine score {cand.score:.4f} {status_str} threshold {self.threshold:.4f}"

            elif strat == "bm25":
                # Normalize BM25 score with query term coverage to avoid direct raw-score comparison
                norm_bm25, cov, bm_reason = compute_bm25_relevance(query, cand.text, cand.score)
                is_rel = norm_bm25 >= self.threshold
                status_str = ">=" if is_rel else "<"
                reason = f"{bm_reason} {status_str} threshold {self.threshold:.4f}"

            elif strat == "hybrid":
                # Evaluates both underlying dense similarity and lexical coverage
                d_score = float(cand.metadata.get("dense_score", cand.score))
                b_score = float(cand.metadata.get("bm25_score", 0.0))
                norm_bm25, cov, bm_reason = compute_bm25_relevance(query, cand.text, b_score)
                dense_pass = d_score >= self.threshold
                bm25_pass = norm_bm25 >= self.threshold
                is_rel = dense_pass or bm25_pass
                status_str = ">=" if is_rel else "<"
                reason = (
                    f"Hybrid candidate (dense={d_score:.4f}, bm25_norm={norm_bm25:.4f} [cov={cov:.2f}]) "
                    f"{status_str} threshold {self.threshold:.4f}"
                )

            else:
                # Default fallback comparison
                is_rel = cand.score >= self.threshold
                reason = f"Score {cand.score:.4f} vs threshold {self.threshold:.4f}"

            decision = GradingDecision(
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                score=cand.score,
                is_relevant=is_rel,
                reasoning=reason,
            )
            decisions.append(decision)

            if is_rel:
                passed.append(cand)
            else:
                rejected.append(cand)

        return GradingResult(
            passed_chunks=passed,
            rejected_chunks=rejected,
            is_sufficient=len(passed) > 0,
            method="threshold",
            threshold=self.threshold,
            decisions=decisions,
        )

    def _grade_with_llm(self, query: str, candidates: list[RetrievalResult]) -> GradingResult:
        passed: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []
        decisions: list[GradingDecision] = []

        assert self.llm_provider is not None

        for cand in candidates:
            prompt = (
                "You are an expert relevance evaluator for a retrieval-augmented generation (RAG) system.\n"
                "Evaluate whether the following retrieved text passage contains information or facts "
                "relevant to answering the user question.\n\n"
                f"User Question: {query}\n\n"
                f"Retrieved Passage: {cand.text}\n\n"
                "Criteria:\n"
                "- Reply 'YES' if the passage contains relevant facts, definitions, context, or evidence.\n"
                "- Reply 'NO' if the passage is irrelevant, unrelated, or contains no useful evidence.\n\n"
                "Answer with ONLY 'YES' or 'NO'."
            )

            try:
                raw_resp = self.llm_provider.generate(prompt)
                clean_resp = raw_resp.strip().upper()
                is_rel = clean_resp.startswith("YES") or "YES" in clean_resp
                reason = f"LLM grader classified passage as: '{clean_resp[:30]}'"
            except ProviderError as pe:
                logger.warning(f"LLM grading failed for chunk {cand.chunk_id}: {pe}. Falling back to threshold.")
                is_rel = cand.score >= self.threshold
                reason = f"LLM grader error ({pe}); fallback to threshold {self.threshold} (score={cand.score:.4f})"
            except Exception as e:
                logger.warning(f"Unexpected error in LLM grading: {e}. Falling back to threshold.")
                is_rel = cand.score >= self.threshold
                reason = f"LLM error ({e}); fallback to threshold {self.threshold} (score={cand.score:.4f})"

            decision = GradingDecision(
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                score=cand.score,
                is_relevant=is_rel,
                reasoning=reason,
            )
            decisions.append(decision)

            if is_rel:
                passed.append(cand)
            else:
                rejected.append(cand)

        return GradingResult(
            passed_chunks=passed,
            rejected_chunks=rejected,
            is_sufficient=len(passed) > 0,
            method="llm",
            threshold=self.threshold,
            decisions=decisions,
        )


# ------------------------------------------------------------------------------
# Functional Interface
# ------------------------------------------------------------------------------

def grade_relevance(
    query: str,
    candidates: list[RetrievalResult],
    method: str = "threshold",
    threshold: float = 0.5,
    llm_provider: Optional[BaseLLMProvider] = None,
) -> GradingResult:
    """
    Public functional interface for evaluating candidate passage relevance.
    """
    grader = RelevanceGrader(method=method, threshold=threshold, llm_provider=llm_provider)
    return grader.grade(query=query, candidates=candidates)
