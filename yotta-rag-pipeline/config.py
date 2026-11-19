"""
Configuration module for the Yotta Workstream A RAG Pipeline.

Loads settings from environment variables and .env files, providing
strongly-typed, validated configurations for LLM generation, embeddings,
and RAG retrieval parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv


def mask_secret(secret: Optional[str], show_prefix: int = 3, show_suffix: int = 4) -> str:
    """Safely mask secrets (API keys) so they are never printed in plain text."""
    if not secret:
        return "(not set)"
    s = secret.strip()
    if len(s) <= (show_prefix + show_suffix + 2):
        return "***"
    return f"{s[:show_prefix]}...{s[-show_suffix:]}"


@dataclass
class LLMConfig:
    """Configuration for LLM response generation."""
    provider: str = "openai"  # openai | nvidia_nim | ollama | mock
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    timeout: int = 30

    def validate(self) -> list[str]:
        """Validate LLM configuration and return any warnings or errors."""
        issues: list[str] = []
        prov = self.provider.lower().strip().replace("-", "_")
        if prov not in {"openai", "nvidia_nim", "ollama", "mock"}:
            issues.append(f"Unsupported LLM provider '{self.provider}'. Choose from: openai, nvidia_nim, ollama, mock.")

        if prov == "openai" and not self.api_key:
            issues.append("OpenAI provider selected, but OPENAI_API_KEY is not set.")

        if prov == "nvidia_nim" and not self.api_key:
            issues.append("NVIDIA NIM provider selected, but NVIDIA_API_KEY is not set.")

        return issues


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model (dense vector retrieval)."""
    provider: str = "sentence_transformers"  # sentence_transformers | openai | nvidia_nim | ollama | mock
    model: str = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30

    def validate(self) -> list[str]:
        """Validate embedding configuration and return any warnings or errors."""
        issues: list[str] = []
        prov = self.provider.lower().strip().replace("-", "_")
        if prov not in {"sentence_transformers", "openai", "nvidia_nim", "ollama", "mock"}:
            issues.append(
                f"Unsupported embedding provider '{self.provider}'. "
                f"Choose from: sentence_transformers, openai, nvidia_nim, ollama, mock."
            )

        if prov == "openai" and not self.api_key:
            issues.append("OpenAI embedding provider selected, but OPENAI_API_KEY is not set.")

        if prov == "nvidia_nim" and not self.api_key:
            issues.append("NVIDIA NIM embedding provider selected, but NVIDIA_API_KEY is not set.")

        return issues


@dataclass
class RAGConfig:
    """Hyperparameters for ingestion, chunking, and retrieval."""
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 3
    relevance_threshold: float = 0.5
    max_retries: int = 1

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.chunk_size <= 0:
            issues.append("chunk_size must be positive.")
        if self.chunk_overlap < 0:
            issues.append("chunk_overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            issues.append("chunk_overlap must be strictly less than chunk_size.")
        if self.top_k <= 0:
            issues.append("top_k must be greater than 0.")
        if not (0.0 <= self.relevance_threshold <= 1.0):
            issues.append("relevance_threshold must be between 0.0 and 1.0.")
        if self.max_retries < 0:
            issues.append("max_retries cannot be negative.")
        return issues


@dataclass
class AppConfig:
    """Root application configuration aggregating all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    log_level: str = "INFO"

    def validate(self) -> list[str]:
        issues = []
        issues.extend(self.llm.validate())
        issues.extend(self.embedding.validate())
        issues.extend(self.rag.validate())
        return issues

    def get_summary(self) -> dict[str, Any]:
        """Return a safe summary dict of active configurations with secrets masked."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": mask_secret(self.llm.api_key),
                "base_url": self.llm.base_url or "(default)",
                "temperature": self.llm.temperature,
                "timeout": self.llm.timeout,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "api_key": mask_secret(self.embedding.api_key),
                "base_url": self.embedding.base_url or "(default)",
                "timeout": self.embedding.timeout,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
                "top_k": self.rag.top_k,
                "relevance_threshold": self.rag.relevance_threshold,
                "max_retries": self.rag.max_retries,
            },
            "log_level": self.log_level,
        }


def load_config(env_path: Optional[str | Path] = None) -> AppConfig:
    """
    Load configuration from environment variables and optional .env file.
    
    Precedence:
    1. System environment variables
    2. .env file in project directory
    3. Default fallbacks
    """
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        # Default lookup in project directory, current working directory, and parent directory
        candidate_paths = [
            Path(__file__).resolve().parent / ".env",
            Path(".env"),
            Path(__file__).resolve().parent.parent / ".env",
        ]
        for p in candidate_paths:
            if p.exists():
                load_dotenv(dotenv_path=p)
                break

    # 1. LLM Config
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower().replace("-", "_")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    
    # Provider-specific keys and base URLs
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-bolt-e4b")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Resolve active LLM credentials and endpoint
    llm_api_key = None
    llm_base_url = None
    if llm_provider == "openai":
        llm_api_key = openai_key
        llm_base_url = openai_base_url
    elif llm_provider == "nvidia_nim":
        llm_api_key = nvidia_key
        llm_base_url = nvidia_base_url
        # If NVIDIA_MODEL is explicitly specified in env, use it; otherwise fallback to configured LLM_MODEL
        if os.getenv("NVIDIA_MODEL"):
            llm_model = os.getenv("NVIDIA_MODEL").strip()
        elif not os.getenv("LLM_MODEL"):
            llm_model = nvidia_model
    elif llm_provider == "ollama":
        llm_base_url = ollama_base_url
        if os.getenv("OLLAMA_MODEL"):
            llm_model = os.getenv("OLLAMA_MODEL").strip()
        elif not os.getenv("LLM_MODEL"):
            llm_model = ollama_model

    llm_config = LLMConfig(
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        timeout=int(os.getenv("LLM_REQUEST_TIMEOUT", "30")),
    )

    # 2. Embedding Config
    embed_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower().replace("-", "_")
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

    embed_api_key = None
    embed_base_url = None
    if embed_provider == "openai":
        embed_api_key = openai_key
        embed_base_url = openai_base_url
        if not os.getenv("EMBEDDING_MODEL"):
            embed_model = "text-embedding-3-small"
    elif embed_provider == "nvidia_nim":
        embed_api_key = nvidia_key
        embed_base_url = nvidia_base_url
        if not os.getenv("EMBEDDING_MODEL"):
            embed_model = "NV-Embed-QA"
    elif embed_provider == "ollama":
        embed_base_url = ollama_base_url
        if not os.getenv("EMBEDDING_MODEL"):
            embed_model = ollama_embed_model

    embed_config = EmbeddingConfig(
        provider=embed_provider,
        model=embed_model,
        api_key=embed_api_key,
        base_url=embed_base_url,
        timeout=int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30")),
    )

    # 3. RAG Config
    rag_config = RAGConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "3")),
        relevance_threshold=float(os.getenv("RELEVANCE_THRESHOLD", "0.5")),
        max_retries=int(os.getenv("MAX_RETRIES", "1")),
    )

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    return AppConfig(
        llm=llm_config,
        embedding=embed_config,
        rag=rag_config,
        log_level=log_level,
    )
