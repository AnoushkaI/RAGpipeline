"""
Retrieval Module for Yotta Workstream A RAG Pipeline.

Implements two high-performance retrieval strategies:
1. DenseRetriever: Vector cosine-similarity search using dot products over
   L2-normalized NumPy embedding matrices.
2. BM25Retriever: Lexical keyword retrieval utilizing rank-bm25 BM25Okapi scores.

Both retrievers return standardized RetrievalResult objects preserving chunk ID,
parent document ID, chunk text, strategy provenance, and ranking scores.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from rag.indexing import BM25Index, DenseIndex, tokenize_text
from rag.providers import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Standard Retrieval Result Model
# ------------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Standardized container for a single retrieved chunk.
    
    Attributes:
        chunk_id: Unique identifier of the retrieved chunk.
        doc_id: Identifier of the parent document/passage.
        text: Text content of the chunk.
        score: Relevance score (cosine similarity in [-1.0, 1.0] for dense,
               BM25Okapi score >= 0.0 for lexical).
        metadata: Chunk and parent document metadata.
        strategy: Retrieval strategy used ('dense' or 'bm25').
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    strategy: str = "dense"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------
# Dense Retriever
# ------------------------------------------------------------------------------

class DenseRetriever:
    """
    Performs dense vector retrieval via cosine similarity over a DenseIndex.
    Uses safe L2 normalization to compute cosine similarity as matrix-vector dot product.
    """

    def __init__(self, index: DenseIndex, embedding_provider: BaseEmbeddingProvider):
        self.index = index
        self.embedding_provider = embedding_provider

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        Retrieves the top-k chunks highest in cosine similarity to the query.
        
        Args:
            query: User search text or question.
            top_k: Number of candidate chunks to return (default: 3).
            
        Returns:
            List of RetrievalResult objects ranked descending by similarity score.
        """
        if self.index.is_empty or top_k <= 0:
            return []

        clean_query = query.strip() if query else ""
        if not clean_query:
            logger.debug("Empty query passed to DenseRetriever. Returning zero-score results.")
            scores = np.zeros(len(self.index), dtype=np.float32)
        else:
            try:
                query_vec = self.embedding_provider.embed([clean_query])[0]
                query_arr = np.array(query_vec, dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to compute query embedding: {e}")
                raise

            if query_arr.size != self.index.dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.index.dimension}, "
                    f"got {query_arr.size}"
                )

            # Safely normalize query vector
            q_norm = float(np.linalg.norm(query_arr))
            if q_norm < 1e-12:
                logger.warning("Query vector has zero or near-zero norm.")
                query_normed = np.zeros_like(query_arr)
            else:
                query_normed = query_arr / q_norm

            # Cosine similarity is the dot product of normalized vectors
            scores = np.dot(self.index.embeddings, query_normed)
            scores = np.nan_to_num(scores, nan=0.0)

        actual_k = min(top_k, len(self.index))
        # Top-k indices sorted descending
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            chunk = self.index.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    metadata=dict(chunk.metadata),
                    strategy="dense",
                )
            )

        return results


# ------------------------------------------------------------------------------
# BM25 Lexical Retriever
# ------------------------------------------------------------------------------

class BM25Retriever:
    """
    Performs lexical keyword retrieval using BM25Okapi over a BM25Index.
    """

    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        Retrieves the top-k chunks highest in BM25 score to the query.
        
        Args:
            query: User search text or question.
            top_k: Number of candidate chunks to return (default: 3).
            
        Returns:
            List of RetrievalResult objects ranked descending by BM25 score.
        """
        if self.index.is_empty or top_k <= 0 or self.index.bm25 is None:
            return []

        clean_query = query.strip() if query else ""
        query_tokens = tokenize_text(clean_query)

        if not query_tokens:
            logger.debug("Query produced zero tokens for BM25. Returning zero-score results.")
            scores = np.zeros(len(self.index), dtype=np.float32)
        else:
            raw_scores = self.index.bm25.get_scores(query_tokens)
            scores = np.array(raw_scores, dtype=np.float32)
            scores = np.nan_to_num(scores, nan=0.0)

        actual_k = min(top_k, len(self.index))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            chunk = self.index.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    metadata=dict(chunk.metadata),
                    strategy="bm25",
                )
            )

        return results


# ------------------------------------------------------------------------------
# Hybrid Retriever (Reciprocal Rank Fusion - RRF)
# ------------------------------------------------------------------------------

class HybridRetriever:
    """
    Performs hybrid retrieval combining dense vector similarity and BM25 lexical ranking
    via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        bm25_index: BM25Index,
        embedding_provider: BaseEmbeddingProvider,
        rrf_k: int = 60,
    ):
        self.dense_retriever = DenseRetriever(index=dense_index, embedding_provider=embedding_provider)
        self.bm25_retriever = BM25Retriever(index=bm25_index)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        Retrieves top-k chunks by fusing rankings from dense and BM25 retrievers.
        Score formula: RRF(d) = sum(1.0 / (rrf_k + rank_m(d))) for m in {dense, bm25}.
        """
        if top_k <= 0:
            return []

        candidate_k = max(top_k * 2, 10)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for rank, res in enumerate(dense_results, start=1):
            chunk_map[res.chunk_id] = res
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, res in enumerate(bm25_results, start=1):
            chunk_map[res.chunk_id] = res
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        dense_map = {r.chunk_id: r for r in dense_results}
        bm25_map = {r.chunk_id: r for r in bm25_results}
        max_possible_rrf = 2.0 / (self.rrf_k + 1)

        results: list[RetrievalResult] = []
        for cid in sorted_chunk_ids:
            orig = chunk_map[cid]
            d_res = dense_map.get(cid)
            b_res = bm25_map.get(cid)
            d_score = float(d_res.score) if d_res is not None else 0.0
            b_score = float(b_res.score) if b_res is not None else 0.0

            meta = dict(orig.metadata)
            meta["dense_score"] = d_score
            meta["bm25_score"] = b_score
            meta["rrf_rank_score"] = float(rrf_scores[cid])

            norm_score = min(1.0, float(rrf_scores[cid] / max_possible_rrf))
            results.append(
                RetrievalResult(
                    chunk_id=orig.chunk_id,
                    doc_id=orig.doc_id,
                    text=orig.text,
                    score=norm_score,
                    metadata=meta,
                    strategy="hybrid",
                )
            )

        return results


# ------------------------------------------------------------------------------
# Unified Top-K Retrieval Entrypoint
# ------------------------------------------------------------------------------

def retrieve_top_k(
    query: str,
    index: DenseIndex | BM25Index | tuple[DenseIndex, BM25Index],
    strategy: Literal["dense", "bm25", "hybrid"] = "dense",
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    bm25_index: Optional[BM25Index] = None,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """
    Unified entrypoint to retrieve top-k chunks from DenseIndex, BM25Index, or Hybrid.
    
    Args:
        query: User search string.
        index: Target index (DenseIndex, BM25Index, or tuple of (DenseIndex, BM25Index)).
        strategy: 'dense', 'bm25', or 'hybrid'.
        embedding_provider: Required if strategy is 'dense' or 'hybrid'.
        bm25_index: Optional BM25Index when strategy is 'hybrid' and index is DenseIndex.
        top_k: Number of top results to return.
        
    Returns:
        Ranked list of RetrievalResult objects.
    """
    strat = strategy.lower().strip()
    if strat == "dense":
        if not isinstance(index, DenseIndex):
            raise TypeError(f"Expected DenseIndex for dense strategy, got {type(index)}")
        if embedding_provider is None:
            raise ValueError("embedding_provider must be supplied for dense retrieval.")
        retriever = DenseRetriever(index=index, embedding_provider=embedding_provider)
        return retriever.retrieve(query=query, top_k=top_k)
    elif strat == "bm25":
        if not isinstance(index, BM25Index):
            raise TypeError(f"Expected BM25Index for bm25 strategy, got {type(index)}")
        retriever = BM25Retriever(index=index)
        return retriever.retrieve(query=query, top_k=top_k)
    elif strat == "hybrid":
        if embedding_provider is None:
            raise ValueError("embedding_provider must be supplied for hybrid retrieval.")
        if isinstance(index, tuple) and len(index) == 2:
            d_idx, b_idx = index
        elif isinstance(index, DenseIndex) and bm25_index is not None:
            d_idx, b_idx = index, bm25_index
        else:
            raise ValueError("Hybrid retrieval requires both DenseIndex and BM25Index.")
        retriever = HybridRetriever(dense_index=d_idx, bm25_index=b_idx, embedding_provider=embedding_provider)
        return retriever.retrieve(query=query, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieval strategy '{strategy}'. Choose 'dense', 'bm25', or 'hybrid'.")
