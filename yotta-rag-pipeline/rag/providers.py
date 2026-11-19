"""
Provider abstraction layer for Yotta Workstream A RAG Pipeline.

Implements unified interfaces for text generation and dense embeddings,
supporting OpenAI, NVIDIA NIM (OpenAI-compatible), Ollama/Llama,
local sentence-transformers, and mock providers for testing.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import requests

from config import AppConfig, EmbeddingConfig, LLMConfig, mask_secret

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Exceptions and Health Data
# ------------------------------------------------------------------------------

class ProviderError(Exception):
    """Base exception for provider failures."""
    pass


class ProviderConfigurationError(ProviderError):
    """Raised when configuration or credentials are missing/invalid."""
    pass


class ProviderConnectionError(ProviderError):
    """Raised when the remote endpoint cannot be reached or times out."""
    pass


class ProviderExecutionError(ProviderError):
    """Raised when the model fails during generation or embedding."""
    pass


@dataclass
class ProviderHealth:
    """Status record for provider connectivity/smoke test."""
    name: str
    kind: str  # 'generation' or 'embedding'
    status: str  # 'OK' | 'FAILED' | 'MISSING_CREDENTIALS' | 'UNTESTED'
    model: str
    latency_ms: Optional[float] = None
    message: str = ""

    def __str__(self) -> str:
        latency_str = f" ({self.latency_ms:.1f}ms)" if self.latency_ms is not None else ""
        return f"[{self.status}] {self.name} ({self.kind} - {self.model}){latency_str}: {self.message}"


# ------------------------------------------------------------------------------
# Base Interfaces
# ------------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base class for LLM text generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response text from a prompt."""
        pass

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Verify connectivity and operational readiness."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for dense text embedding."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute vector embeddings for a list of input texts."""
        pass

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Verify connectivity and operational readiness."""
        pass


class ModelProvider:
    """
    Unified provider facade combining generation and embedding capabilities.
    Allows the pipeline to interact with a single interface while keeping
    generation and embedding configs decoupled.
    """

    def __init__(self, llm: BaseLLMProvider, embedding: BaseEmbeddingProvider):
        self.llm = llm
        self.embedding = embedding

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedding.embed(texts)

    def run_health_checks(self) -> dict[str, ProviderHealth]:
        return {
            "llm": self.llm.health_check(),
            "embedding": self.embedding.health_check(),
        }


# ------------------------------------------------------------------------------
# Mock Providers (For Deterministic Testing & Offline Development)
# ------------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock LLM for testing without network/API costs."""

    def __init__(self, model: str = "mock-model", default_response: str = "This is a mock grounded answer."):
        self.model = model
        self.default_response = default_response

    def generate(self, prompt: str) -> str:
        return self.default_response

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            name="MockLLM",
            kind="generation",
            status="OK",
            model=self.model,
            latency_ms=0.5,
            message="Mock LLM ready for testing.",
        )


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding provider returning fixed-dimension vectors."""

    def __init__(self, dimension: int = 8, model: str = "mock-embed"):
        self.dimension = dimension
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Deterministic pseudo-embedding based on string length and ascii codes
        embeddings: list[list[float]] = []
        for text in texts:
            val = float(len(text))
            vec = [(val + i) / (val + 10.0) for i in range(self.dimension)]
            # normalize approximately
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            name="MockEmbedding",
            kind="embedding",
            status="OK",
            model=self.model,
            latency_ms=0.5,
            message="Mock Embedding ready for testing.",
        )


# ------------------------------------------------------------------------------
# OpenAI Provider
# ------------------------------------------------------------------------------

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI generation provider using openai-python client."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def _get_client(self):
        if not self.config.api_key:
            raise ProviderConfigurationError("OpenAI API key is missing. Set OPENAI_API_KEY in .env.")
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.openai.com/v1",
                timeout=self.config.timeout,
            )
        except Exception as e:
            raise ProviderConfigurationError(f"Failed to initialize OpenAI client: {e}") from e

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise ProviderExecutionError(f"OpenAI generation failed: {e}") from e

    def health_check(self) -> ProviderHealth:
        if not self.config.api_key:
            return ProviderHealth(
                name="OpenAI",
                kind="generation",
                status="MISSING_CREDENTIALS",
                model=self.config.model,
                message="OPENAI_API_KEY is not set.",
            )
        start_t = time.perf_counter()
        try:
            client = self._get_client()
            client.models.list()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="OpenAI",
                kind="generation",
                status="OK",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message="Successfully authenticated and reached OpenAI endpoint.",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="OpenAI",
                kind="generation",
                status="FAILED",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Connection failed: {str(e)}",
            )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using openai-python client."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def _get_client(self):
        if not self.config.api_key:
            raise ProviderConfigurationError("OpenAI API key is missing. Set OPENAI_API_KEY in .env.")
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.openai.com/v1",
                timeout=self.config.timeout,
            )
        except Exception as e:
            raise ProviderConfigurationError(f"Failed to initialize OpenAI client: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            response = client.embeddings.create(
                input=texts,
                model=self.config.model,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise ProviderExecutionError(f"OpenAI embedding failed: {e}") from e

    def health_check(self) -> ProviderHealth:
        if not self.config.api_key:
            return ProviderHealth(
                name="OpenAI",
                kind="embedding",
                status="MISSING_CREDENTIALS",
                model=self.config.model,
                message="OPENAI_API_KEY is not set.",
            )
        start_t = time.perf_counter()
        try:
            self.embed(["ping"])
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="OpenAI",
                kind="embedding",
                status="OK",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message="Successfully generated test embedding.",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="OpenAI",
                kind="embedding",
                status="FAILED",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Embedding check failed: {str(e)}",
            )


# ------------------------------------------------------------------------------
# NVIDIA NIM Provider (OpenAI-compatible)
# ------------------------------------------------------------------------------

class NvidiaNimLLMProvider(BaseLLMProvider):
    """
    NVIDIA NIM generation provider via OpenAI-compatible API.
    Supports official chat completions, Nemotron 3.5 Bolt E4B,
    and standalone gateway staging prefixes ('stg/').
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._resolved_model: Optional[str] = None

    def _sanitize_error(self, err: Exception | str) -> str:
        """Mask any API keys, tokens, or auth headers from error output."""
        import re
        msg = str(err)
        if self.config.api_key:
            msg = msg.replace(self.config.api_key, "***")
        msg = re.sub(r'nvapi-[A-Za-z0-9_-]+', 'nvapi-***', msg)
        msg = re.sub(r'Bearer\s+[A-Za-z0-9_.-]+', 'Bearer ***', msg)
        return msg

    def _candidate_models(self) -> list[str]:
        """
        Return candidate model identifiers to support standard catalog
        and standalone gateway staging prefixes (e.g., 'stg/nvidia/nemotron-3.5-bolt-e4b').
        """
        primary = (self._resolved_model or self.config.model).strip()
        candidates = [primary]
        if primary.startswith("stg/"):
            candidates.append(primary[4:])
        else:
            candidates.append(f"stg/{primary}")
        return candidates

    def _get_client(self):
        if not self.config.api_key:
            raise ProviderConfigurationError("NVIDIA API key is missing. Set NVIDIA_API_KEY in .env.")
        try:
            import openai
            base_url = self.config.base_url or "https://integrate.api.nvidia.com/v1"
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=base_url,
                timeout=self.config.timeout,
            )
        except Exception as e:
            raise ProviderConfigurationError(
                f"Failed to initialize NVIDIA NIM client: {self._sanitize_error(e)}"
            ) from e

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        last_error = None

        for model_id in self._candidate_models():
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                )
                self._resolved_model = model_id
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                err_text = str(e).lower()
                # If 404 or not found, try the alternate candidate prefix
                if "404" in err_text or "not found" in err_text:
                    continue
                raise ProviderExecutionError(
                    f"NVIDIA NIM generation failed on model '{model_id}': {self._sanitize_error(e)}"
                ) from e

        raise ProviderExecutionError(
            f"NVIDIA NIM generation failed for '{self.config.model}' and alternate prefixes: {self._sanitize_error(last_error)}"
        )

    def health_check(self) -> ProviderHealth:
        if not self.config.api_key:
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="generation",
                status="MISSING_CREDENTIALS",
                model=self.config.model,
                message="NVIDIA_API_KEY is not set in environment or .env.",
            )
        start_t = time.perf_counter()
        try:
            client = self._get_client()
            last_err = None

            for model_id in self._candidate_models():
                try:
                    # Verify actual model generation with a minimal test prompt
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": "Respond with 'OK'"}],
                        max_tokens=10,
                        temperature=0.0,
                    )
                    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                    content = (resp.choices[0].message.content or "").strip()
                    self._resolved_model = model_id
                    return ProviderHealth(
                        name="NVIDIA NIM",
                        kind="generation",
                        status="OK",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        message=f"Verified live generation via chat completions API. Sample output: '{content[:30]}'",
                    )
                except Exception as e:
                    last_err = e
                    err_text = str(e).lower()
                    if "404" in err_text or "not found" in err_text:
                        continue
                    break

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            safe_msg = self._sanitize_error(last_err)
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="generation",
                status="FAILED",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"NVIDIA NIM generation check failed: {safe_msg}",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="generation",
                status="FAILED",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"NVIDIA NIM initialization check failed: {self._sanitize_error(e)}",
            )


class NvidiaNimEmbeddingProvider(BaseEmbeddingProvider):
    """
    NVIDIA NIM embedding provider via OpenAI-compatible endpoint.
    Maintains strict separation between LLM generation models and dedicated embedding models.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def _sanitize_error(self, err: Exception | str) -> str:
        import re
        msg = str(err)
        if self.config.api_key:
            msg = msg.replace(self.config.api_key, "***")
        msg = re.sub(r'nvapi-[A-Za-z0-9_-]+', 'nvapi-***', msg)
        msg = re.sub(r'Bearer\s+[A-Za-z0-9_.-]+', 'Bearer ***', msg)
        return msg

    def _validate_embedding_model(self):
        """Prevent generation models (like Nemotron Bolt) from being erroneously used as embeddings."""
        m = self.config.model.lower()
        if "bolt" in m or "instruct" in m or "llama-3.1-8b" in m or "chat" in m:
            raise ProviderConfigurationError(
                f"'{self.config.model}' is an LLM text generation model, not an embedding model. "
                "For dense retrieval, use the default 'sentence_transformers' provider with 'all-MiniLM-L6-v2', "
                "or configure a dedicated NVIDIA embedding model such as 'nvidia/nemotron-3.5-embed-1b' or 'NV-Embed-QA'."
            )

    def _get_client(self):
        if not self.config.api_key:
            raise ProviderConfigurationError("NVIDIA API key is missing. Set NVIDIA_API_KEY in .env.")
        self._validate_embedding_model()
        try:
            import openai
            base_url = self.config.base_url or "https://integrate.api.nvidia.com/v1"
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=base_url,
                timeout=self.config.timeout,
            )
        except Exception as e:
            raise ProviderConfigurationError(
                f"Failed to initialize NVIDIA NIM client: {self._sanitize_error(e)}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            response = client.embeddings.create(
                input=texts,
                model=self.config.model,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise ProviderExecutionError(f"NVIDIA NIM embedding failed: {self._sanitize_error(e)}") from e

    def health_check(self) -> ProviderHealth:
        if not self.config.api_key:
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="embedding",
                status="MISSING_CREDENTIALS",
                model=self.config.model,
                message="NVIDIA_API_KEY is not set.",
            )
        try:
            self._validate_embedding_model()
        except ProviderConfigurationError as e:
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="embedding",
                status="FAILED",
                model=self.config.model,
                message=str(e),
            )
        start_t = time.perf_counter()
        try:
            embs = self.embed(["health check probe"])
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            dim = len(embs[0]) if embs and embs[0] else 0
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="embedding",
                status="OK",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Successfully generated NVIDIA NIM embedding (dim={dim}).",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="NVIDIA NIM",
                kind="embedding",
                status="FAILED",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"NVIDIA NIM embedding check failed: {self._sanitize_error(e)}",
            )


# ------------------------------------------------------------------------------
# Llama / Ollama Provider
# ------------------------------------------------------------------------------

class OllamaLLMProvider(BaseLLMProvider):
    """Llama generation provider via local or hosted Ollama REST endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            ) from e
        except Exception as e:
            raise ProviderExecutionError(f"Ollama generation failed: {e}") from e

    def health_check(self) -> ProviderHealth:
        start_t = time.perf_counter()
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            installed_models = [m.get("name", "") for m in tags]
            
            # Verify configured model is installed (exact, with tag, or base name)
            target = self.config.model
            model_installed = any(
                target == name or name.startswith(f"{target}:") or target == name.split(":")[0]
                for name in installed_models
            )
            
            if not model_installed:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                installed_str = ", ".join(installed_models) or "none"
                return ProviderHealth(
                    name="Ollama (Llama)",
                    kind="generation",
                    status="FAILED",
                    model=self.config.model,
                    latency_ms=elapsed_ms,
                    message=(
                        f"Ollama server reachable, but model '{self.config.model}' is not installed. "
                        f"Installed models: [{installed_str}]. Run `ollama pull {self.config.model}`."
                    ),
                )

            # Test actual generation with minimal prompt
            reply = self.generate("Say OK")
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="Ollama (Llama)",
                kind="generation",
                status="OK",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Model verified and functional. Response snippet: '{reply.strip()[:20]}'",
            )
        except requests.exceptions.ConnectionError:
            return ProviderHealth(
                name="Ollama (Llama)",
                kind="generation",
                status="FAILED",
                model=self.config.model,
                message=f"Cannot reach Ollama at {self.base_url}. Ensure Ollama daemon is running.",
            )
        except Exception as e:
            return ProviderHealth(
                name="Ollama (Llama)",
                kind="generation",
                status="FAILED",
                model=self.config.model,
                message=f"Ollama health check failed: {str(e)}",
            )


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Llama/Ollama dense embedding provider via modern /api/embed endpoint,
    supporting batched input with fallback to legacy /api/embeddings.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # 1. Primary: Modern Ollama /api/embed endpoint with batched input
        url_embed = f"{self.base_url}/api/embed"
        payload = {"model": self.config.model, "input": texts}
        try:
            resp = requests.post(url_embed, json=payload, timeout=self.config.timeout)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) == len(texts):
                    return embeddings
            elif resp.status_code != 404:
                resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot reach Ollama embedding service at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            ) from e
        except requests.exceptions.RequestException as e:
            logger.debug(f"/api/embed request failed ({e}), attempting legacy /api/embeddings fallback.")

        # 2. Legacy fallback to /api/embeddings (single item iteration)
        url_legacy = f"{self.base_url}/api/embeddings"
        embeddings: list[list[float]] = []
        for text in texts:
            try:
                resp = requests.post(url_legacy, json={"model": self.config.model, "prompt": text}, timeout=self.config.timeout)
                resp.raise_for_status()
                emb = resp.json().get("embedding", [])
                embeddings.append(emb)
            except requests.exceptions.ConnectionError as e:
                raise ProviderConnectionError(
                    f"Cannot reach Ollama embedding service at {self.base_url}."
                ) from e
            except Exception as e:
                raise ProviderExecutionError(f"Ollama embedding failed for text: {e}") from e
        return embeddings

    def health_check(self) -> ProviderHealth:
        start_t = time.perf_counter()
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            installed_models = [m.get("name", "") for m in tags]
            
            # Verify configured embedding model is actually installed
            target = self.config.model
            model_installed = any(
                target == name or name.startswith(f"{target}:") or target == name.split(":")[0]
                for name in installed_models
            )
            
            if not model_installed:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                installed_str = ", ".join(installed_models) or "none"
                return ProviderHealth(
                    name="Ollama (Embeddings)",
                    kind="embedding",
                    status="FAILED",
                    model=self.config.model,
                    latency_ms=elapsed_ms,
                    message=(
                        f"Ollama daemon is reachable, but configured embedding model '{self.config.model}' "
                        f"is not installed. Installed models: [{installed_str}]. "
                        f"Run `ollama pull {self.config.model}`."
                    ),
                )
            
            # Model is confirmed installed; test live embedding generation
            test_embs = self.embed(["health check probe"])
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            dim = len(test_embs[0]) if test_embs and test_embs[0] else 0
            return ProviderHealth(
                name="Ollama (Embeddings)",
                kind="embedding",
                status="OK",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Model verified and functional via /api/embed (vector dim={dim}).",
            )
        except requests.exceptions.ConnectionError:
            return ProviderHealth(
                name="Ollama (Embeddings)",
                kind="embedding",
                status="FAILED",
                model=self.config.model,
                message=f"Cannot reach Ollama at {self.base_url}. Ensure Ollama daemon is running.",
            )
        except Exception as e:
            return ProviderHealth(
                name="Ollama (Embeddings)",
                kind="embedding",
                status="FAILED",
                model=self.config.model,
                message=f"Ollama embedding check failed: {str(e)}",
            )


# ------------------------------------------------------------------------------
# SentenceTransformers Local Provider (Zero-Cost Local Baseline)
# ------------------------------------------------------------------------------

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local dense embedding provider using sentence-transformers.
    Requires no API keys, runs completely locally on CPU/GPU.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model_instance = None

    def _resolve_local_cache_path(self, model_name: str) -> Optional[str]:
        """Check if model snapshot is cached in ~/.cache/huggingface/hub to enable fast offline loading."""
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.exists():
            return None
        
        candidates = [model_name]
        if "/" in model_name:
            candidates.append(model_name.replace("/", "--"))
        else:
            candidates.append(f"sentence-transformers--{model_name}")

        for cand in candidates:
            folder_name = cand if cand.startswith("models--") else f"models--{cand}"
            snapshots_dir = cache_dir / folder_name / "snapshots"
            if snapshots_dir.exists():
                snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
                if snapshots:
                    return str(snapshots[0])
        return None

    def _get_model(self):
        if self._model_instance is None:
            try:
                import sys
                print(f"  [i] Initializing local SentenceTransformer '{self.config.model}'...", flush=True)
                from sentence_transformers import SentenceTransformer
                
                # Check for cached snapshot first for instantaneous offline resolution
                cached_path = self._resolve_local_cache_path(self.config.model)
                if cached_path:
                    self._model_instance = SentenceTransformer(cached_path)
                else:
                    self._model_instance = SentenceTransformer(self.config.model)
            except Exception as e:
                raise ProviderExecutionError(
                    f"Failed to load SentenceTransformer model '{self.config.model}': {e}"
                ) from e
        return self._model_instance

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            raise ProviderExecutionError(f"SentenceTransformer encoding failed: {e}") from e

    def health_check(self) -> ProviderHealth:
        start_t = time.perf_counter()
        try:
            emb = self.embed(["health check probe"])
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            dim = len(emb[0]) if emb else 0
            return ProviderHealth(
                name="SentenceTransformers (Local)",
                kind="embedding",
                status="OK",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Model loaded locally successfully (vector dim={dim}).",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ProviderHealth(
                name="SentenceTransformers (Local)",
                kind="embedding",
                status="FAILED",
                model=self.config.model,
                latency_ms=elapsed_ms,
                message=f"Local model load/embed failed: {str(e)}",
            )


# ------------------------------------------------------------------------------
# Provider Factory Functions
# ------------------------------------------------------------------------------

def get_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """Instantiate the configured LLM generation provider."""
    provider_type = config.provider.lower().strip().replace("-", "_")
    if provider_type == "openai":
        return OpenAILLMProvider(config)
    elif provider_type == "nvidia_nim":
        return NvidiaNimLLMProvider(config)
    elif provider_type == "ollama":
        return OllamaLLMProvider(config)
    elif provider_type == "mock":
        return MockLLMProvider(model=config.model)
    else:
        raise ProviderConfigurationError(
            f"Unsupported LLM provider '{config.provider}'. "
            f"Available options: openai, nvidia_nim, ollama, mock."
        )


def get_embedding_provider(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """Instantiate the configured embedding provider."""
    provider_type = config.provider.lower().strip().replace("-", "_")
    if provider_type == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(config)
    elif provider_type == "openai":
        return OpenAIEmbeddingProvider(config)
    elif provider_type == "nvidia_nim":
        return NvidiaNimEmbeddingProvider(config)
    elif provider_type == "ollama":
        return OllamaEmbeddingProvider(config)
    elif provider_type == "mock":
        return MockEmbeddingProvider(model=config.model)
    else:
        raise ProviderConfigurationError(
            f"Unsupported embedding provider '{config.provider}'. "
            f"Available options: sentence_transformers, openai, nvidia_nim, ollama, mock."
        )


def get_model_provider(config: AppConfig) -> ModelProvider:
    """Create unified ModelProvider from full AppConfig."""
    llm = get_llm_provider(config.llm)
    emb = get_embedding_provider(config.embedding)
    return ModelProvider(llm=llm, embedding=emb)


# ------------------------------------------------------------------------------
# Smoke Test Runner CLI
# ------------------------------------------------------------------------------

def run_provider_smoke_test(config: Optional[AppConfig] = None) -> int:
    """
    Run diagnostic connectivity smoke tests for configured generation and embedding providers.
    Reports honest status without leaking secrets.
    Returns 0 if all active checks pass, 1 if any active check fails.
    """
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    from config import load_config
    cfg = config or load_config()

    print("=" * 70)
    print(" YOTTA RAG PIPELINE - PROVIDER CONNECTIVITY & SMOKE TEST")
    print("=" * 70)
    print(f"Generation Provider: {cfg.llm.provider} (Model: {cfg.llm.model})")
    print(f"Embedding Provider:  {cfg.embedding.provider} (Model: {cfg.embedding.model})")
    print("-" * 70)

    # Validate configuration warnings first
    issues = cfg.validate()
    if issues:
        print("Configuration notices / warnings:")
        for issue in issues:
            print(f"  [!] {issue}")
        print("-" * 70)

    llm_provider = get_llm_provider(cfg.llm)
    emb_provider = get_embedding_provider(cfg.embedding)

    print("Testing Generation Provider...")
    llm_health = llm_provider.health_check()
    print(f"  {llm_health}")

    print("\nTesting Embedding Provider...")
    emb_health = emb_provider.health_check()
    print(f"  {emb_health}")

    print("=" * 70)
    
    # Check overall status
    has_failure = (llm_health.status == "FAILED" or emb_health.status == "FAILED")
    has_missing_creds = (llm_health.status == "MISSING_CREDENTIALS" or emb_health.status == "MISSING_CREDENTIALS")
    has_untested = (llm_health.status == "UNTESTED" or emb_health.status == "UNTESTED")

    if has_failure:
        print("RESULT: FAILED — One or more providers failed health checks or are missing configured models.")
        return 1
    elif has_missing_creds:
        print("RESULT: INCOMPLETE — Credentials are missing for one or more configured providers.")
        print("        (Configure API keys in .env or switch to mock/sentence_transformers mode).")
        return 1
    elif has_untested:
        print("RESULT: INCOMPLETE — One or more providers are configured but untested.")
        return 1
    else:
        print("RESULT: SUCCESS — All configured providers passed live health checks.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_provider_smoke_test())
