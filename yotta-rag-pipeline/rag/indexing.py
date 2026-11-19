"""
Indexing Module for Yotta Workstream A RAG Pipeline.

Implements two primary indexing structures:
1. DenseIndex: In-memory NumPy matrix of L2-normalized dense embeddings,
   retaining chunk IDs, parent document IDs, text, and metadata.
2. BM25Index: Inverted lexical keyword index powered by rank-bm25 (BM25Okapi)
   over the exact same chunks and tokenization scheme.

Designed for high efficiency, lightweight memory footprint, and exact
reproducibility across CPU/GPU environments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.providers import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def tokenize_text(text: str) -> list[str]:
    """
    Standard lexical tokenizer for BM25 indexing and query processing.
    Converts to lowercase and extracts alphanumeric word tokens.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def safe_l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Safely L2-normalizes each row of a 2D float NumPy matrix in-place or returns normalized copy.
    Replaces near-zero norms with 1.0 to prevent division by zero or NaN creation.
    """
    if matrix.size == 0:
        return matrix
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix for normalization, got shape {matrix.shape}")
    
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms < eps, 1.0, norms)
    return matrix / safe_norms


# ------------------------------------------------------------------------------
# Dense Index Structure
# ------------------------------------------------------------------------------

@dataclass
class DenseIndex:
    """
    In-memory dense vector index backed by a 2D NumPy array.
    
    Attributes:
        embeddings: 2D NumPy array of shape (N, D), where each row is L2-normalized.
        chunks: List of Chunk objects aligned with the matrix rows.
        chunk_ids: List of chunk ID strings.
        doc_ids: List of parent document ID strings.
        dimension: Embedding dimension D.
    """
    embeddings: np.ndarray
    chunks: list[Chunk]
    chunk_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    dimension: int = 0

    def __post_init__(self):
        if not self.chunk_ids and self.chunks:
            self.chunk_ids = [c.chunk_id for c in self.chunks]
        if not self.doc_ids and self.chunks:
            self.doc_ids = [c.doc_id for c in self.chunks]
        if self.embeddings.ndim == 2:
            self.dimension = self.embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0 or self.embeddings.size == 0


# ------------------------------------------------------------------------------
# BM25 Index Structure
# ------------------------------------------------------------------------------

@dataclass
class BM25Index:
    """
    Lexical keyword index backed by BM25Okapi over the same chunk corpus.
    
    Attributes:
        bm25: The BM25Okapi instance from rank-bm25.
        chunks: List of Chunk objects aligned with the BM25 documents.
        chunk_ids: List of chunk ID strings.
        doc_ids: List of parent document ID strings.
        tokenized_corpus: List of token lists for every indexed chunk.
    """
    bm25: Optional[BM25Okapi]
    chunks: list[Chunk]
    chunk_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    tokenized_corpus: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.chunk_ids and self.chunks:
            self.chunk_ids = [c.chunk_id for c in self.chunks]
        if not self.doc_ids and self.chunks:
            self.doc_ids = [c.doc_id for c in self.chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0 or self.bm25 is None


# ------------------------------------------------------------------------------
# Index Builders
# ------------------------------------------------------------------------------

def build_dense_index(
    chunks: list[Chunk],
    embedding_provider: BaseEmbeddingProvider,
    batch_size: int = 64,
) -> DenseIndex:
    """
    Embeds all chunks using the provided embedding provider, stores vectors in a
    2D NumPy array, normalizes vectors safely for cosine similarity, and maps
    chunk IDs, parent doc IDs, and metadata.
    
    Args:
        chunks: List of Chunk instances to index.
        embedding_provider: Working BaseEmbeddingProvider (e.g. SentenceTransformer).
        batch_size: Batch size for embedding calls.
        
    Returns:
        DenseIndex containing normalized embeddings and metadata.
    """
    if not chunks:
        logger.warning("Empty chunks list passed to build_dense_index. Creating empty index.")
        empty_matrix = np.empty((0, 0), dtype=np.float32)
        return DenseIndex(embeddings=empty_matrix, chunks=[], dimension=0)

    logger.info(f"Building dense index for {len(chunks)} chunks with batch_size={batch_size}...")
    texts = [c.text for c in chunks]

    # Batch embedding to respect memory and API limits
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = embedding_provider.embed(batch)
        all_embeddings.extend(batch_vecs)

    if not all_embeddings:
        raise ValueError("Embedding provider returned empty embeddings for chunks.")

    raw_matrix = np.array(all_embeddings, dtype=np.float32)
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding matrix shape mismatch: expected ({len(chunks)}, D), "
            f"got {raw_matrix.shape}"
        )

    # Safely L2-normalize for rapid cosine similarity dot-products
    normalized_matrix = safe_l2_normalize(raw_matrix)
    dimension = int(normalized_matrix.shape[1])

    logger.info(
        f"Dense index built successfully: {len(chunks)} chunks, dimension={dimension}, "
        f"memory_est={normalized_matrix.nbytes / 1024:.2f} KB."
    )

    return DenseIndex(
        embeddings=normalized_matrix,
        chunks=chunks,
        chunk_ids=[c.chunk_id for c in chunks],
        doc_ids=[c.doc_id for c in chunks],
        dimension=dimension,
    )


def build_bm25_index(
    chunks: list[Chunk],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """
    Builds a rank-bm25 BM25Okapi lexical index over the exact same chunks.
    
    Args:
        chunks: List of Chunk instances to index.
        k1: BM25 term frequency saturation parameter (default: 1.5).
        b: BM25 document length normalization parameter (default: 0.75).
        
    Returns:
        BM25Index containing the BM25Okapi index and tokenized corpus.
    """
    if not chunks:
        logger.warning("Empty chunks list passed to build_bm25_index. Creating empty index.")
        return BM25Index(bm25=None, chunks=[], tokenized_corpus=[])

    logger.info(f"Building BM25 index for {len(chunks)} chunks (k1={k1}, b={b})...")
    tokenized_corpus = [tokenize_text(c.text) for c in chunks]

    # Handle case where all tokens in all chunks are empty
    non_empty = any(len(tokens) > 0 for tokens in tokenized_corpus)
    if not non_empty:
        logger.warning("All chunk tokens were empty. Initializing minimal dummy token.")
        tokenized_corpus = [["<empty>"] for _ in chunks]

    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    logger.info(f"BM25 index built successfully for {len(chunks)} chunks.")
    return BM25Index(
        bm25=bm25,
        chunks=chunks,
        chunk_ids=[c.chunk_id for c in chunks],
        doc_ids=[c.doc_id for c in chunks],
        tokenized_corpus=tokenized_corpus,
    )


def build_indices(
    chunks: list[Chunk],
    embedding_provider: BaseEmbeddingProvider,
    batch_size: int = 64,
) -> tuple[DenseIndex, BM25Index]:
    """
    Unified entrypoint to construct both Dense and BM25 indices simultaneously
    over the exact same chunks.
    
    Args:
        chunks: List of Chunk objects.
        embedding_provider: Configured embedding provider.
        batch_size: Batch size for dense embedding.
        
    Returns:
        Tuple of (DenseIndex, BM25Index).
    """
    dense_idx = build_dense_index(chunks, embedding_provider=embedding_provider, batch_size=batch_size)
    bm25_idx = build_bm25_index(chunks)
    return dense_idx, bm25_idx
