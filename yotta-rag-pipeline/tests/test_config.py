"""
Unit tests for configuration loading, validation, and secret masking.
"""

from __future__ import annotations

import os
import pytest
from config import AppConfig, EmbeddingConfig, LLMConfig, RAGConfig, load_config, mask_secret


def test_mask_secret():
    """Verify secrets are never revealed in plain text."""
    assert mask_secret(None) == "(not set)"
    assert mask_secret("") == "(not set)"
    assert mask_secret("short") == "***"
    masked = mask_secret("sk-proj-abcdefghijklmnopqrstuvwxyz1234")
    assert masked.startswith("sk-")
    assert masked.endswith("1234")
    assert "..." in masked
    assert "abcdef" not in masked


def test_default_config():
    """Verify sensible defaults are populated."""
    cfg = AppConfig()
    assert cfg.llm.provider == "openai"
    assert cfg.embedding.provider == "sentence_transformers"
    assert cfg.rag.chunk_size == 300
    assert cfg.rag.chunk_overlap == 50
    assert cfg.rag.top_k == 3
    assert cfg.rag.max_retries == 1


def test_config_validation():
    """Verify validation flags configuration mistakes without crashing."""
    # Invalid provider
    cfg = AppConfig(llm=LLMConfig(provider="unsupported_provider"))
    issues = cfg.validate()
    assert any("Unsupported LLM provider" in msg for msg in issues)

    # Overlap >= chunk size
    cfg_overlap = AppConfig(rag=RAGConfig(chunk_size=100, chunk_overlap=150))
    issues_overlap = cfg_overlap.validate()
    assert any("chunk_overlap must be strictly less than chunk_size" in msg for msg in issues_overlap)

    # Missing API keys for OpenAI / NIM
    cfg_openai = AppConfig(llm=LLMConfig(provider="openai", api_key=None))
    assert any("OPENAI_API_KEY is not set" in msg for msg in cfg_openai.validate())


def test_load_config_env_override(monkeypatch):
    """Verify environment variables properly override defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("TOP_K", "5")

    cfg = load_config()
    assert cfg.llm.provider == "mock"
    assert cfg.embedding.provider == "mock"
    assert cfg.rag.chunk_size == 500
    assert cfg.rag.top_k == 5
