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
# Unified Top-K Retrieval Entrypoint
# ------------------------------------------------------------------------------

def retrieve_top_k(
    query: str,
    index: DenseIndex | BM25Index,
    strategy: Literal["dense", "bm25"] = "dense",
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """
    Unified entrypoint to retrieve top-k chunks from either DenseIndex or BM25Index.
    
    Args:
        query: User search string.
        index: Target index (DenseIndex or BM25Index).
        strategy: 'dense' or 'bm25'.
        embedding_provider: Required if strategy is 'dense'.
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
    else:
        raise ValueError(f"Unknown retrieval strategy '{strategy}'. Choose 'dense' or 'bm25'.")
