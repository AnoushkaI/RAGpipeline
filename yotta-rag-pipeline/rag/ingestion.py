"""
Data Ingestion & Cleaning Module for Yotta Workstream A RAG Pipeline.

Responsible for:
- Loading reproducible MS MARCO subsets from Hugging Face.
- Loading local JSONL fallback passages for 100% offline reliability.
- Text cleaning, unicode normalization, control character removal, whitespace stripping.
- Deduplication via content hashing.
- Schema validation and persistent metadata / doc_id attachment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Core Data Models
# ------------------------------------------------------------------------------

@dataclass
class Passage:
    """
    Represents an ingested, cleaned document passage.
    
    Attributes:
        doc_id: Unique, stable identifier for the passage.
        text: Normalized and cleaned passage text.
        metadata: Associated source information, title, domain, etc.
    """
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Passage:
        return cls(
            doc_id=str(data.get("doc_id", "")).strip(),
            text=str(data.get("text", "")).strip(),
            metadata=data.get("metadata", {}),
        )


# ------------------------------------------------------------------------------
# Text Cleaning & Normalization
# ------------------------------------------------------------------------------

def clean_passage_text(text: str) -> str:
    """
    Clean, normalize, and sanitize raw passage text.
    
    Operations:
    1. NFKC unicode normalization (standardizes accents, curly quotes, ligatures).
    2. Strips unprintable control characters (preserving standard newlines/tabs).
    3. Normalizes multiple spaces, tabs, and duplicate newlines.
    4. Trims leading and trailing whitespace.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    normalized = unicodedata.normalize("NFKC", str(text))

    # 2. Remove non-printable control characters except standard space and newline
    cleaned = "".join(
        ch for ch in normalized
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " ")
    )

    # 3. Collapse multiple whitespace/tabs to single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 4. Collapse 3+ newlines to max 2 newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text for deduplication."""
    canonical = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------
# Cleaning, Deduplication & Validation Pipeline
# ------------------------------------------------------------------------------

def clean_and_validate_passages(
    raw_passages: list[dict[str, Any] | Passage],
    min_length: int = 20,
    deduplicate: bool = True,
) -> list[Passage]:
    """
    Processes a list of raw passages: cleans text, filters short/empty items,
    deduplicates identical text, and ensures stable IDs and metadata.
    
    Args:
        raw_passages: Raw passage dictionaries or Passage objects.
        min_length: Minimum character length to retain (filters junk/stubs).
        deduplicate: Whether to filter identical text content.
        
    Returns:
        List of validated, cleaned, unique Passage instances.
    """
    cleaned_passages: list[Passage] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_passages):
        if isinstance(item, Passage):
            doc_id = item.doc_id
            text = item.text
            metadata = dict(item.metadata)
        elif isinstance(item, dict):
            doc_id = str(item.get("doc_id", f"doc_{idx:04d}")).strip()
            text = str(item.get("text", "")).strip()
            metadata = dict(item.get("metadata", {}))
        else:
            continue

        # Clean text
        cleaned_text = clean_passage_text(text)

        # Filter out empty or trivially short passages
        if len(cleaned_text) < min_length:
            continue

        # Deduplication by content hash
        if deduplicate:
            content_hash = compute_content_hash(cleaned_text)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

        # Guarantee unique doc_id
        if not doc_id or doc_id in seen_ids:
            doc_id = f"doc_{len(cleaned_passages) + 1:04d}"
        seen_ids.add(doc_id)

        # Preserve metadata and record character count
        metadata["char_count"] = len(cleaned_text)
        metadata["word_count"] = len(cleaned_text.split())

        cleaned_passages.append(
            Passage(doc_id=doc_id, text=cleaned_text, metadata=metadata)
        )

    return cleaned_passages


# ------------------------------------------------------------------------------
# Data Loaders (Local Fallback & Hugging Face MS MARCO)
# ------------------------------------------------------------------------------

def load_local_fallback(
    filepath: Optional[str | Path] = None,
    min_length: int = 20,
) -> list[Passage]:
    """
    Loads passages from local JSONL fallback file.
    Ensures the RAG pipeline remains fully operational offline or without HF access.
    """
    if filepath is None:
        # Default project data path
        base_dir = Path(__file__).resolve().parent.parent / "data"
        filepath = base_dir / "sample_passages.jsonl"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Local fallback dataset not found at: {filepath}")

    raw_items: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                raw_items.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON on line {line_num} in {filepath}: {e}")

    return clean_and_validate_passages(raw_items, min_length=min_length, deduplicate=True)


def load_msmarco_subset(
    sample_size: int = 100,
    split: str = "train",
    seed: int = 42,
    fallback_on_error: bool = True,
) -> list[Passage]:
    """
    Loads a reproducible small subset of MS MARCO from Hugging Face Datasets.
    
    If network access, dataset revision, or environment prevents downloading,
    it automatically falls back to `load_local_fallback()`.
    
    Args:
        sample_size: Number of passages to extract (default: 100).
        split: Hugging Face dataset split (e.g., 'train').
        seed: Random seed for deterministic subset selection.
        fallback_on_error: If True, falls back to local JSONL when HF fails.
    """
    logger.info(f"Attempting to load MS MARCO subset (size={sample_size}, split={split})...")
    try:
        from datasets import load_dataset

        # Load small slice deterministically
        dataset = load_dataset(
            "ms_marco",
            "v1.1",
            split=f"{split}[:{sample_size * 2}]",  # load 2x to account for cleaning/dedup
            trust_remote_code=False,
        )

        raw_items: list[dict[str, Any]] = []
        for idx, row in enumerate(dataset):
            # MS MARCO v1.1 structure: 'passages': {'passage_text': [...], 'is_selected': [...]}
            passages_data = row.get("passages", {})
            passage_texts = passages_data.get("passage_text", [])
            query_id = row.get("query_id", idx)

            for p_idx, text in enumerate(passage_texts):
                doc_id = f"msmarco_{query_id}_{p_idx}"
                raw_items.append({
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": {
                        "source": "hf_msmarco_v1.1",
                        "query_id": query_id,
                        "query_type": row.get("query_type", "unknown"),
                    },
                })
                if len(raw_items) >= sample_size:
                    break
            if len(raw_items) >= sample_size:
                break

        passages = clean_and_validate_passages(raw_items, min_length=20, deduplicate=True)
        return passages[:sample_size]

    except Exception as e:
        logger.warning(
            f"Hugging Face MS MARCO download failed or unavailable ({e}). "
            "Falling back to local curated JSONL dataset."
        )
        if fallback_on_error:
            return load_local_fallback()
        raise


def load_corpus(
    use_hf: bool = False,
    sample_size: int = 100,
    local_path: Optional[str | Path] = None,
) -> list[Passage]:
    """
    Unified entrypoint to load the passage corpus for the RAG pipeline.
    
    Args:
        use_hf: If True, attempts Hugging Face MS MARCO with fallback.
                If False, directly loads curated local fallback dataset.
        sample_size: Maximum passages to load.
        local_path: Custom local JSONL file path if specified.
    """
    if use_hf:
        return load_msmarco_subset(sample_size=sample_size, fallback_on_error=True)
    return load_local_fallback(filepath=local_path)
