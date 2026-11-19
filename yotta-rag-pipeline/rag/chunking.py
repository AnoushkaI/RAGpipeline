"""
Chunking Strategies Module for Yotta Workstream A RAG Pipeline.

Implements two distinct chunking strategies:
1. FixedSizeChunker: Fixed character window with sliding overlap.
2. SentenceBoundaryChunker: Splits on grammatical sentence boundaries without severing clauses.

Both strategies maintain strict parent passage ID lineage (`chunk_id = f"{doc_id}_c{idx}"`)
and inherit parent document metadata for citations and retrieval evaluation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rag.ingestion import Passage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Core Chunk Model
# ------------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    Represents an indexed chunk derived from a parent Passage.
    
    Attributes:
        chunk_id: Unique chunk identifier (e.g. 'doc_0001_f0').
        doc_id: Identifier of the parent Passage from which this chunk was created.
        text: Text content of the chunk.
        chunk_index: 0-indexed position within parent document.
        strategy: Chunking strategy used ('fixed' or 'sentence').
        metadata: Inherited metadata plus chunk-specific metrics.
    """
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    strategy: str  # 'fixed' | 'sentence'
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------
# Fixed-Size Chunker with Overlap
# ------------------------------------------------------------------------------

class FixedSizeChunker:
    """
    Splits passage text into fixed character windows with sliding overlap.
    Preserves whole words where possible by snapping window boundaries to spaces.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, passage: Passage) -> list[Chunk]:
        text = passage.text.strip()
        if not text:
            return []

        # If text is shorter than or equal to chunk_size, return single chunk
        if len(text) <= self.chunk_size:
            meta = dict(passage.metadata)
            meta.update({
                "char_count": len(text),
                "word_count": len(text.split()),
                "start_char": 0,
                "end_char": len(text),
            })
            return [
                Chunk(
                    chunk_id=f"{passage.doc_id}_f0",
                    doc_id=passage.doc_id,
                    text=text,
                    chunk_index=0,
                    strategy="fixed",
                    metadata=meta,
                )
            ]

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Snap to word boundary if not at the absolute end of the text
            if end < text_len:
                # Look backwards up to 30 chars for a space
                space_pos = text.rfind(" ", start + (self.chunk_size // 2), end)
                if space_pos != -1:
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = dict(passage.metadata)
                meta.update({
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "start_char": start,
                    "end_char": end,
                })
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage.doc_id}_f{chunk_idx}",
                        doc_id=passage.doc_id,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        strategy="fixed",
                        metadata=meta,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break

            # Slide window forward by step
            start = end - self.chunk_overlap
            # Avoid infinite loop if start does not advance
            if start <= chunks[-1].metadata.get("start_char", 0):
                start = end

        return chunks


# ------------------------------------------------------------------------------
# Sentence-Boundary Chunker
# ------------------------------------------------------------------------------

class SentenceBoundaryChunker:
    """
    Alternative chunking strategy that splits text into coherent sentence groups.
    Ensures complete sentences are preserved and never severed mid-thought.
    """

    # Regex matching sentence terminal punctuation (. ? !) followed by whitespace or end of string
    SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"\'\(\[])')

    def __init__(self, target_chunk_size: int = 300, max_chunk_size: int = 450):
        if target_chunk_size <= 0:
            raise ValueError(f"target_chunk_size must be positive, got {target_chunk_size}")
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences cleanly."""
        text = text.strip()
        if not text:
            return []
        # Split on sentence boundaries
        raw_sentences = self.SENTENCE_SPLIT_REGEX.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def chunk(self, passage: Passage) -> list[Chunk]:
        text = passage.text.strip()
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_len = len(sentence)
            # If adding this sentence exceeds target_chunk_size and we already have sentences
            if current_sentences and (current_len + sent_len + 1 > self.target_chunk_size):
                chunk_text = " ".join(current_sentences).strip()
                meta = dict(passage.metadata)
                meta.update({
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "sentence_count": len(current_sentences),
                })
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage.doc_id}_s{chunk_idx}",
                        doc_id=passage.doc_id,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        strategy="sentence",
                        metadata=meta,
                    )
                )
                chunk_idx += 1
                current_sentences = [sentence]
                current_len = sent_len
            else:
                current_sentences.append(sentence)
                current_len += sent_len + 1

        # Emit remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            meta = dict(passage.metadata)
            meta.update({
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "sentence_count": len(current_sentences),
            })
            chunks.append(
                Chunk(
                    chunk_id=f"{passage.doc_id}_s{chunk_idx}",
                    doc_id=passage.doc_id,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    strategy="sentence",
                    metadata=meta,
                )
            )

        return chunks


# ------------------------------------------------------------------------------
# Unified Chunking API
# ------------------------------------------------------------------------------

def chunk_document(
    passage: Passage,
    strategy: Literal["fixed", "sentence"] = "fixed",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Chunks a single document using the specified strategy.
    
    Args:
        passage: The Passage to chunk.
        strategy: 'fixed' (fixed character window with overlap) or 'sentence' (sentence boundaries).
        chunk_size: Window size in characters for 'fixed', or target size for 'sentence'.
        chunk_overlap: Sliding window overlap in characters for 'fixed'.
    """
    strat = strategy.lower().strip()
    if strat == "fixed":
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunker.chunk(passage)
    elif strat == "sentence":
        chunker = SentenceBoundaryChunker(target_chunk_size=chunk_size)
        return chunker.chunk(passage)
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose 'fixed' or 'sentence'.")


def chunk_corpus(
    passages: list[Passage],
    strategy: Literal["fixed", "sentence"] = "fixed",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Chunks an entire corpus of passages, preserving stable document IDs and metadata.
    
    Args:
        passages: List of Passage objects.
        strategy: 'fixed' or 'sentence'.
        chunk_size: Configurable chunk size.
        chunk_overlap: Configurable chunk overlap.
        
    Returns:
        List of all generated Chunk instances across the corpus.
    """
    all_chunks: list[Chunk] = []
    for passage in passages:
        doc_chunks = chunk_document(
            passage=passage,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
