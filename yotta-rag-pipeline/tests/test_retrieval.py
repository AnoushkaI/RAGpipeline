"""
Targeted tests for Dense & BM25 Retrieval (rag/retrieval.py).
Tests ranking logic, edge cases, zero vectors, and metadata preservation.
"""

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.indexing import build_bm25_index, build_dense_index
from rag.providers import BaseEmbeddingProvider, MockEmbeddingProvider, ProviderHealth
from rag.retrieval import BM25Retriever, DenseRetriever, RetrievalResult, retrieve_top_k


class ControlledEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider that returns pre-determined vectors for testing ranking order.
    """
    def __init__(self, mapping: dict[str, list[float]], default_vec: list[float]):
        self.mapping = mapping
        self.default_vec = default_vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping.get(t, self.default_vec) for t in texts]

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(name="Controlled", kind="embedding", status="OK", model="test")


@pytest.fixture
def chunk_corpus() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_001_f0",
            doc_id="doc_001",
            text="Gradient descent optimizes loss functions.",
            chunk_index=0,
            strategy="fixed",
            metadata={"topic": "optimization"},
        ),
        Chunk(
            chunk_id="doc_002_f0",
            doc_id="doc_002",
            text="B-Trees reduce disk I/O in relational databases.",
            chunk_index=0,
            strategy="fixed",
            metadata={"topic": "databases"},
        ),
        Chunk(
            chunk_id="doc_003_f0",
            doc_id="doc_003",
            text="Docker containers provide process isolation on Linux.",
            chunk_index=0,
            strategy="fixed",
            metadata={"topic": "devops"},
        ),
    ]


def test_dense_retrieval_ranking(chunk_corpus):
    # Vector dim = 3.
    # Chunk 0 has high affinity with [1, 0, 0]
    # Chunk 1 has high affinity with [0, 1, 0]
    # Chunk 2 has high affinity with [0, 0, 1]
    provider = ControlledEmbeddingProvider(
        mapping={
            chunk_corpus[0].text: [1.0, 0.0, 0.0],
            chunk_corpus[1].text: [0.0, 1.0, 0.0],
            chunk_corpus[2].text: [0.0, 0.0, 1.0],
            "query databases": [0.05, 0.95, 0.0],  # Close to chunk 1
        },
        default_vec=[0.0, 0.0, 0.0],
    )

    dense_idx = build_dense_index(chunk_corpus, embedding_provider=provider)
    retriever = DenseRetriever(index=dense_idx, embedding_provider=provider)

    results = retriever.retrieve(query="query databases", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    # Most relevant must be doc_002
    assert results[0].doc_id == "doc_002"
    assert results[0].chunk_id == "doc_002_f0"
    assert results[0].strategy == "dense"
    assert results[0].score > results[1].score
    assert results[0].metadata["topic"] == "databases"


def test_dense_retrieval_empty_query_or_index(chunk_corpus):
    mock_provider = MockEmbeddingProvider(dimension=4)
    dense_idx = build_dense_index(chunk_corpus, embedding_provider=mock_provider)
    retriever = DenseRetriever(index=dense_idx, embedding_provider=mock_provider)

    # Empty query string
    empty_results = retriever.retrieve(query="", top_k=2)
    assert len(empty_results) == 2
    # Should not crash, scores should be 0.0
    assert all(r.score == 0.0 for r in empty_results)

    # top_k <= 0
    assert retriever.retrieve(query="valid query", top_k=0) == []

    # Empty index
    empty_idx = build_dense_index([], embedding_provider=mock_provider)
    empty_retriever = DenseRetriever(index=empty_idx, embedding_provider=mock_provider)
    assert empty_retriever.retrieve(query="valid query", top_k=3) == []


def test_dense_retrieval_dimension_mismatch(chunk_corpus):
    mock_provider_4 = MockEmbeddingProvider(dimension=4)
    dense_idx = build_dense_index(chunk_corpus, embedding_provider=mock_provider_4)

    # Query provider returns dim 8
    mock_provider_8 = MockEmbeddingProvider(dimension=8)
    retriever = DenseRetriever(index=dense_idx, embedding_provider=mock_provider_8)

    with pytest.raises(ValueError, match="dimension mismatch"):
        retriever.retrieve(query="test", top_k=2)


def test_bm25_retrieval_ranking(chunk_corpus):
    bm25_idx = build_bm25_index(chunk_corpus)
    retriever = BM25Retriever(index=bm25_idx)

    # Query matching doc_003 specifically
    results = retriever.retrieve(query="docker containers linux isolation", top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc_003"
    assert results[0].chunk_id == "doc_003_f0"
    assert results[0].strategy == "bm25"
    assert results[0].score > 0.0
    assert results[0].metadata["topic"] == "devops"


def test_bm25_retrieval_empty_query_or_index(chunk_corpus):
    bm25_idx = build_bm25_index(chunk_corpus)
    retriever = BM25Retriever(index=bm25_idx)

    empty_results = retriever.retrieve(query="", top_k=2)
    assert len(empty_results) == 2
    assert all(r.score == 0.0 for r in empty_results)

    assert retriever.retrieve(query="containers", top_k=0) == []


def test_retrieve_top_k_facade(chunk_corpus):
    mock_provider = MockEmbeddingProvider(dimension=4)
    dense_idx = build_dense_index(chunk_corpus, embedding_provider=mock_provider)
    bm25_idx = build_bm25_index(chunk_corpus)

    # Dense via facade
    dense_res = retrieve_top_k(
        query="test query",
        index=dense_idx,
        strategy="dense",
        embedding_provider=mock_provider,
        top_k=1,
    )
    assert len(dense_res) == 1
    assert dense_res[0].strategy == "dense"

    # BM25 via facade
    bm25_res = retrieve_top_k(
        query="containers",
        index=bm25_idx,
        strategy="bm25",
        top_k=1,
    )
    assert len(bm25_res) == 1
    assert bm25_res[0].strategy == "bm25"
