"""
Unit tests for Fixed-Size and Sentence-Boundary Chunking Strategies.
"""

from __future__ import annotations

import pytest
from rag.chunking import (
    Chunk,
    FixedSizeChunker,
    SentenceBoundaryChunker,
    chunk_corpus,
    chunk_document,
)
from rag.ingestion import Passage


@pytest.fixture
def sample_passage() -> Passage:
    long_text = (
        "Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. "
        "The data, known as training data, consists of a set of training examples. "
        "Each training example has one or more inputs and a desired output value, also known as a supervisory signal. "
        "In contrast, unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data, "
        "like grouping or clustering of data points. "
        "The algorithms therefore learn from test data that has not been labeled, classified or categorized. "
        "Semi-supervised learning falls in between, using a small amount of labeled data and a large amount of unlabeled data."
    )
    return Passage(
        doc_id="doc_test_01",
        text=long_text,
        metadata={"domain": "machine_learning", "author": "Yotta AI"},
    )


def test_fixed_size_chunking_basic(sample_passage):
    """Verify fixed-size chunking creates overlapping chunks with stable IDs."""
    chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(sample_passage)

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == sample_passage.doc_id for c in chunks)
    assert all(c.strategy == "fixed" for c in chunks)
    assert chunks[0].chunk_id == "doc_test_01_f0"
    assert chunks[1].chunk_id == "doc_test_01_f1"

    # Verify metadata inheritance
    assert chunks[0].metadata["domain"] == "machine_learning"
    assert chunks[0].metadata["author"] == "Yotta AI"
    assert "char_count" in chunks[0].metadata
    assert "start_char" in chunks[0].metadata


def test_fixed_size_short_passage():
    """Verify text shorter than chunk_size yields exactly one chunk."""
    short = Passage(doc_id="doc_short", text="Short text under chunk size.")
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk(short)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc_short_f0"
    assert chunks[0].text == "Short text under chunk size."


def test_fixed_size_invalid_params():
    """Verify validation errors on non-positive chunk_size or excessive overlap."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        FixedSizeChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)


def test_sentence_boundary_chunking_basic(sample_passage):
    """Verify sentence-boundary chunker groups sentences without mid-sentence cuts."""
    chunker = SentenceBoundaryChunker(target_chunk_size=200)
    chunks = chunker.chunk(sample_passage)

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == sample_passage.doc_id for c in chunks)
    assert all(c.strategy == "sentence" for c in chunks)
    assert chunks[0].chunk_id == "doc_test_01_s0"

    # Verify each chunk ends with a sentence terminator
    for c in chunks:
        assert c.text[-1] in (".", "?", "!", '"', "'")


def test_sentence_boundary_short_text():
    """Verify sentence chunker handles single-sentence or short text."""
    p = Passage(doc_id="doc_single", text="Just one single sentence here.")
    chunker = SentenceBoundaryChunker(target_chunk_size=300)
    chunks = chunker.chunk(p)

    assert len(chunks) == 1
    assert chunks[0].text == "Just one single sentence here."


def test_chunk_document_api(sample_passage):
    """Verify unified chunk_document function dispatches to both strategies."""
    fixed_chunks = chunk_document(sample_passage, strategy="fixed", chunk_size=200, chunk_overlap=40)
    assert len(fixed_chunks) >= 1
    assert fixed_chunks[0].strategy == "fixed"

    sent_chunks = chunk_document(sample_passage, strategy="sentence", chunk_size=200)
    assert len(sent_chunks) >= 1
    assert sent_chunks[0].strategy == "sentence"

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(sample_passage, strategy="invalid_strategy")  # type: ignore


def test_chunk_corpus():
    """Verify chunk_corpus processes multiple passages and maintains unique chunk IDs."""
    passages = [
        Passage(doc_id="doc_a", text="First passage sentence one. First passage sentence two."),
        Passage(doc_id="doc_b", text="Second passage sentence one. Second passage sentence two."),
    ]
    all_chunks = chunk_corpus(passages, strategy="fixed", chunk_size=40, chunk_overlap=10)

    assert len(all_chunks) >= 2
    chunk_ids = [c.chunk_id for c in all_chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs across corpus must be unique"
    assert any(c.doc_id == "doc_a" for c in all_chunks)
    assert any(c.doc_id == "doc_b" for c in all_chunks)


def test_empty_passage_handling():
    """Verify empty or whitespace passages return empty chunk lists."""
    empty_p = Passage(doc_id="doc_empty", text="   ")
    assert FixedSizeChunker(100, 20).chunk(empty_p) == []
    assert SentenceBoundaryChunker(100).chunk(empty_p) == []
