"""
Targeted tests for Dense & BM25 Indexing (rag/indexing.py).
Uses MockEmbeddingProvider for fast, offline, deterministic execution.
"""

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.indexing import (
    BM25Index,
    DenseIndex,
    build_bm25_index,
    build_dense_index,
    build_indices,
    safe_l2_normalize,
    tokenize_text,
)
from rag.providers import MockEmbeddingProvider


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_001_f0",
            doc_id="doc_001",
            text="Supervised learning uses labeled training datasets for classification.",
            chunk_index=0,
            strategy="fixed",
            metadata={"domain": "ml"},
        ),
        Chunk(
            chunk_id="doc_001_f1",
            doc_id="doc_001",
            text="Unsupervised learning identifies clusters without pre-existing labels.",
            chunk_index=1,
            strategy="fixed",
            metadata={"domain": "ml"},
        ),
        Chunk(
            chunk_id="doc_002_f0",
            doc_id="doc_002",
            text="Overfitting is prevented by regularization techniques like L1 and L2.",
            chunk_index=0,
            strategy="fixed",
            metadata={"domain": "ml"},
        ),
    ]


def test_tokenize_text():
    tokens = tokenize_text("BM25 & TF-IDF: Information Retrieval 101!")
    assert "bm25" in tokens
    assert "tf" in tokens
    assert "idf" in tokens
    assert "information" in tokens
    assert "retrieval" in tokens
    assert "101" in tokens

    # Empty text
    assert tokenize_text("") == []
    assert tokenize_text("   ") == []


def test_safe_l2_normalize():
    # Regular 2D vector
    mat = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normed = safe_l2_normalize(mat)
    # Check that norm of each row is 1.0
    row_norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(row_norms, [1.0, 1.0], atol=1e-5)

    # Zero vector handling (should not produce NaN or error)
    zero_mat = np.array([[0.0, 0.0]], dtype=np.float32)
    normed_zero = safe_l2_normalize(zero_mat)
    assert not np.isnan(normed_zero).any()
    np.testing.assert_allclose(normed_zero, [[0.0, 0.0]])

    # Invalid dimension check
    with pytest.raises(ValueError):
        safe_l2_normalize(np.array([1.0, 2.0]))


def test_build_dense_index(sample_chunks):
    dim = 16
    mock_provider = MockEmbeddingProvider(dimension=dim)
    index = build_dense_index(sample_chunks, embedding_provider=mock_provider, batch_size=2)

    assert isinstance(index, DenseIndex)
    assert len(index) == 3
    assert not index.is_empty
    assert index.dimension == dim
    assert index.embeddings.shape == (3, dim)

    # Verify rows are L2 normalized
    norms = np.linalg.norm(index.embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    # Lineage and metadata preservation
    assert index.chunk_ids == ["doc_001_f0", "doc_001_f1", "doc_002_f0"]
    assert index.doc_ids == ["doc_001", "doc_001", "doc_002"]
    assert index.chunks[0].metadata["domain"] == "ml"


def test_build_dense_index_empty():
    mock_provider = MockEmbeddingProvider(dimension=8)
    empty_idx = build_dense_index([], embedding_provider=mock_provider)
    assert len(empty_idx) == 0
    assert empty_idx.is_empty
    assert empty_idx.embeddings.shape == (0, 0)


def test_build_bm25_index(sample_chunks):
    bm25_idx = build_bm25_index(sample_chunks, k1=1.5, b=0.75)
    assert isinstance(bm25_idx, BM25Index)
    assert len(bm25_idx) == 3
    assert not bm25_idx.is_empty
    assert bm25_idx.chunk_ids == ["doc_001_f0", "doc_001_f1", "doc_002_f0"]
    assert bm25_idx.doc_ids == ["doc_001", "doc_001", "doc_002"]

    # Verify scores for matching token
    scores = bm25_idx.bm25.get_scores(["supervised"])
    assert len(scores) == 3
    assert scores[0] > scores[2]  # doc_001_f0 mentions 'supervised'


def test_build_bm25_index_empty():
    empty_idx = build_bm25_index([])
    assert len(empty_idx) == 0
    assert empty_idx.is_empty
    assert empty_idx.bm25 is None


def test_build_indices_unified(sample_chunks):
    mock_provider = MockEmbeddingProvider(dimension=12)
    dense_idx, bm25_idx = build_indices(sample_chunks, embedding_provider=mock_provider)
    assert len(dense_idx) == 3
    assert len(bm25_idx) == 3
    assert dense_idx.chunk_ids == bm25_idx.chunk_ids
