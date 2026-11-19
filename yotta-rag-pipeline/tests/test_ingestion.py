"""
Unit tests for Data Ingestion, Cleaning, Deduplication, and Validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from rag.ingestion import (
    Passage,
    clean_and_validate_passages,
    clean_passage_text,
    compute_content_hash,
    load_corpus,
    load_local_fallback,
    load_msmarco_subset,
)


def test_clean_passage_text():
    """Verify whitespace stripping, unicode normalization, and control char removal."""
    raw = "  Hello \t  world!\n\n\nThis is a    test\u00A0with special “quotes” and \x00control chars.  "
    cleaned = clean_passage_text(raw)
    assert cleaned.startswith("Hello world!")
    assert "special \"quotes\"" in cleaned or "special “quotes”" in cleaned
    assert "\x00" not in cleaned
    assert "\t" not in cleaned
    assert "    " not in cleaned


def test_compute_content_hash():
    """Verify deterministic hash insensitive to case and extraneous spaces."""
    h1 = compute_content_hash("Machine Learning is Great")
    h2 = compute_content_hash("  machine   learning is great  \n")
    assert h1 == h2


def test_clean_and_validate_passages():
    """Verify validation, short passage rejection, and deduplication."""
    raw_data = [
        {"doc_id": "d1", "text": "Supervised learning models require labeled data for training.", "metadata": {"topic": "ml"}},
        {"doc_id": "d2", "text": "Too short", "metadata": {}},  # shorter than min_length (20)
        {"doc_id": "d3", "text": "   Supervised learning models require labeled data for training.   "},  # duplicate of d1
        {"doc_id": "", "text": "Unsupervised learning discovers intrinsic groupings without supervisory signals."},
    ]

    passages = clean_and_validate_passages(raw_data, min_length=20, deduplicate=True)
    assert len(passages) == 2
    assert passages[0].doc_id == "d1"
    assert passages[0].metadata["topic"] == "ml"
    assert "char_count" in passages[0].metadata
    # Auto-assigned unique ID for empty doc_id
    assert passages[1].doc_id != ""


def test_load_local_fallback():
    """Verify curated fallback dataset loads and conforms to Passage schema."""
    passages = load_local_fallback()
    assert len(passages) >= 20
    assert all(isinstance(p, Passage) for p in passages)
    assert all(p.doc_id.startswith("doc_") for p in passages)
    assert all(len(p.text) > 30 for p in passages)

    # Verify key domains are present
    doc_ids = {p.doc_id for p in passages}
    assert "doc_001" in doc_ids  # Supervised vs Unsupervised
    assert "doc_006" in doc_ids  # BM25


def test_msmarco_fallback_on_error(monkeypatch):
    """Verify load_msmarco_subset falls back cleanly to local fallback when HF dataset is unavailable."""
    # Simulate network failure immediately to test fallback without socket timeout delay
    def fake_load_dataset(*args, **kwargs):
        raise ConnectionError("Simulated network timeout")

    try:
        import datasets
        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    except ImportError:
        pass

    passages = load_msmarco_subset(sample_size=10, fallback_on_error=True)
    assert len(passages) >= 10
    assert all(isinstance(p, Passage) for p in passages)
    assert all(p.doc_id.startswith("doc_") for p in passages)


def test_eval_queries_file():
    """Verify evaluation queries JSON file exists, is valid, and matches corpus IDs."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    eval_file = data_dir / "eval_queries.json"
    assert eval_file.exists()

    with open(eval_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    assert len(queries) >= 10
    passages = load_local_fallback()
    corpus_ids = {p.doc_id for p in passages}

    for item in queries:
        assert "query_id" in item
        assert "query" in item
        assert "relevant_doc_ids" in item
        assert len(item["relevant_doc_ids"]) > 0
        # Check that ground-truth relevant IDs exist in the fallback corpus
        for doc_id in item["relevant_doc_ids"]:
            assert doc_id in corpus_ids, f"Relevant ID '{doc_id}' for query '{item['query_id']}' not in corpus!"
