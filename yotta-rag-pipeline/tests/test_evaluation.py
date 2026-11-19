"""
Targeted tests for Evaluation Metrics & Benchmark (rag/evaluation.py).
Verifies Recall@K calculation, safeguards against double-counting chunks from the same doc,
and provides a local SentenceTransformer integration test.
"""

import pytest

from rag.chunking import Chunk
from rag.evaluation import (
    EvaluationReport,
    calculate_recall_at_k,
    evaluate_retrievers,
    load_eval_queries,
)
from rag.indexing import build_bm25_index, build_dense_index
from rag.providers import EmbeddingConfig, MockEmbeddingProvider, SentenceTransformerEmbeddingProvider
from rag.retrieval import BM25Retriever, DenseRetriever, RetrievalResult


def test_calculate_recall_at_k_single_hit():
    results = [
        RetrievalResult(chunk_id="doc_001_f0", doc_id="doc_001", text="text", score=0.9),
        RetrievalResult(chunk_id="doc_002_f0", doc_id="doc_002", text="text", score=0.7),
        RetrievalResult(chunk_id="doc_003_f0", doc_id="doc_003", text="text", score=0.5),
    ]

    # Target is doc_001
    assert calculate_recall_at_k(results, relevant_doc_ids=["doc_001"], k=1) == 1.0
    # Target is doc_002 at k=1 (miss) and k=2 (hit)
    assert calculate_recall_at_k(results, relevant_doc_ids=["doc_002"], k=1) == 0.0
    assert calculate_recall_at_k(results, relevant_doc_ids=["doc_002"], k=2) == 1.0


def test_calculate_recall_at_k_multiple_chunks_same_doc_no_double_counting():
    """
    CRITICAL REQUIREMENT:
    If multiple chunks from the same parent doc are in top-K, that document
    must NOT be counted more than once.
    """
    # 3 chunks in top-3, all belonging to doc_001
    results = [
        RetrievalResult(chunk_id="doc_001_f0", doc_id="doc_001", text="text", score=0.9),
        RetrievalResult(chunk_id="doc_001_f1", doc_id="doc_001", text="text", score=0.8),
        RetrievalResult(chunk_id="doc_001_f2", doc_id="doc_001", text="text", score=0.7),
    ]

    # Target has 2 documents: doc_001 and doc_002
    # Since only doc_001 was retrieved, Recall@3 must be 1 / 2 = 0.5, NEVER 3 / 2 or 1.0
    recall = calculate_recall_at_k(results, relevant_doc_ids=["doc_001", "doc_002"], k=3)
    assert recall == 0.5


def test_calculate_recall_at_k_edge_cases():
    results = [
        RetrievalResult(chunk_id="doc_001_f0", doc_id="doc_001", text="text", score=0.9),
    ]
    # k <= 0
    assert calculate_recall_at_k(results, relevant_doc_ids=["doc_001"], k=0) == 0.0
    # empty relevant
    assert calculate_recall_at_k(results, relevant_doc_ids=[], k=3) == 0.0
    # empty results
    assert calculate_recall_at_k([], relevant_doc_ids=["doc_001"], k=3) == 0.0


def test_load_eval_queries():
    queries = load_eval_queries()
    assert len(queries) >= 15
    for q in queries:
        assert "query_id" in q
        assert "query" in q
        assert "relevant_doc_ids" in q
        assert len(q["relevant_doc_ids"]) > 0


def test_evaluate_retrievers_mock_benchmark():
    chunks = [
        Chunk("doc_001_f0", "doc_001", "Neural network backpropagation", 0, "fixed"),
        Chunk("doc_002_f0", "doc_002", "Relational database transaction ACID", 0, "fixed"),
    ]
    mock_provider = MockEmbeddingProvider(dimension=4)
    dense_idx = build_dense_index(chunks, mock_provider)
    bm25_idx = build_bm25_index(chunks)

    dense_retriever = DenseRetriever(dense_idx, mock_provider)
    bm25_retriever = BM25Retriever(bm25_idx)

    mock_queries = [
        {"query_id": "q1", "query": "database ACID", "relevant_doc_ids": ["doc_002"]},
        {"query_id": "q2", "query": "backpropagation", "relevant_doc_ids": ["doc_001"]},
    ]

    report = evaluate_retrievers(
        queries=mock_queries,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        k_values=[1, 2],
    )

    assert isinstance(report, EvaluationReport)
    assert report.num_queries == 2
    assert 1 in report.dense_mean_recall
    assert 2 in report.bm25_mean_recall
    summary_table = report.format_summary_table()
    assert "RETRIEVAL EVALUATION REPORT" in summary_table


def test_evaluation_report_export_csv(tmp_path):
    """Verify that EvaluationReport.export_csv writes valid CSV output."""
    chunks = [
        Chunk("doc_001_f0", "doc_001", "Backpropagation computes gradients.", 0, "fixed"),
    ]
    mock_provider = MockEmbeddingProvider(dimension=4)
    dense_idx = build_dense_index(chunks, mock_provider)
    bm25_idx = build_bm25_index(chunks)

    dense_retriever = DenseRetriever(dense_idx, mock_provider)
    bm25_retriever = BM25Retriever(bm25_idx)

    mock_queries = [
        {"query_id": "q1", "query": "gradients", "relevant_doc_ids": ["doc_001"], "category": "ml"},
    ]

    report = evaluate_retrievers(
        queries=mock_queries,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        k_values=[1, 3],
    )

    out_file = tmp_path / "results" / "eval.csv"
    saved_path = report.export_csv(out_file)
    assert saved_path.exists()
    content = saved_path.read_text(encoding="utf-8")
    assert "query_id,query,category,relevant_doc_ids" in content
    assert "q1,gradients,ml,doc_001" in content



def test_local_sentence_transformer_integration():
    """
    Integration test running real SentenceTransformer embeddings on CPU
    for end-to-end indexing and retrieval verification.
    """
    config = EmbeddingConfig(provider="sentence_transformers", model="all-MiniLM-L6-v2")
    provider = SentenceTransformerEmbeddingProvider(config)

    chunks = [
        Chunk("doc_001_f0", "doc_001", "Supervised machine learning algorithms learn from labeled data.", 0, "fixed"),
        Chunk("doc_002_f0", "doc_002", "B-Tree data structures allow efficient disk lookups in database storage engines.", 0, "fixed"),
    ]

    dense_idx = build_dense_index(chunks, provider)
    assert dense_idx.dimension == 384
    assert len(dense_idx) == 2

    retriever = DenseRetriever(dense_idx, provider)
    results = retriever.retrieve("What is supervised machine learning?", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "doc_001"
    assert results[0].score > 0.4
