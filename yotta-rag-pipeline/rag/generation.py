"""
Grounded Generation Module for Yotta Workstream A RAG Pipeline.

Phase 4 Implementation:
- Constructs strictly grounded prompts enforcing reliance solely on retrieved context.
- Requires explicit parent document ID citations in bracketed format (e.g. [doc_0001]).
- Enforces single consolidated answer generation synthesizing facts across multiple chunks.
- Features explicit abstention when retrieved evidence is absent or insufficient.
- Extracts and validates citations against the active context corpus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from rag.providers import BaseLLMProvider, ProviderError
from rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_ABSTENTION_MESSAGE = (
    "I cannot answer this question based on the provided documents "
    "as the necessary information is not present in the retrieved context."
)


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Standardized result for grounded LLM answer generation.
    
    Attributes:
        query: User question submitted.
        answer: Generated single consolidated response text.
        cited_doc_ids: Validated list of parent document IDs cited in the answer.
        context_used: The formatted context string supplied to the LLM.
        has_sufficient_context: True if the answer is grounded; False if abstained.
        raw_prompt: Complete prompt sent to the LLM.
    """
    query: str
    answer: str
    cited_doc_ids: list[str] = field(default_factory=list)
    context_used: str = ""
    has_sufficient_context: bool = True
    raw_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------
# Grounded Generator
# ------------------------------------------------------------------------------

class GroundedGenerator:
    """
    Generates single consolidated answers grounded strictly in retrieved context.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        abstention_message: str = DEFAULT_ABSTENTION_MESSAGE,
    ):
        self.llm_provider = llm_provider
        self.abstention_message = abstention_message

    def build_context_string(self, chunks: list[RetrievalResult]) -> str:
        """
        Formats retrieved chunks with parent document IDs and chunk IDs.
        """
        if not chunks:
            return ""

        context_blocks: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            block = (
                f"--- Context Passage {i} ---\n"
                f"Source Document ID: [{chunk.doc_id}]\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Content: {chunk.text.strip()}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def extract_citations(self, text: str, valid_doc_ids: set[str]) -> list[str]:
        """
        Extracts document citations formatted as [doc_id] from the answer,
        retaining only those that match active source document IDs.
        Preserves order of first appearance and removes duplicates.
        """
        if not text or not valid_doc_ids:
            return []

        raw_matches = re.findall(r"\[([a-zA-Z0-9_\-\.]+)\]", text)
        cited: list[str] = []
        seen: set[str] = set()

        for doc_id in raw_matches:
            if doc_id in valid_doc_ids and doc_id not in seen:
                seen.add(doc_id)
                cited.append(doc_id)

        return cited

    def build_prompt(self, query: str, context_str: str) -> str:
        """
        Constructs the strict grounding prompt for the LLM.
        """
        prompt = (
            "You are a truthful, grounded question-answering assistant.\n"
            "Answer the user's question STRICTLY and ONLY using the facts present in the retrieved context passages below.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Rely EXCLUSIVELY on the facts directly stated in the context passages. Do NOT extrapolate or use outside knowledge.\n"
            "2. Synthesize ONE single cohesive, well-written consolidated answer. Do not output multiple disconnected snippets.\n"
            "3. For every key fact or claim in your answer, you MUST cite the source document ID using brackets, for example: [doc_0001].\n"
            "4. If the provided context does NOT contain sufficient evidence to answer the question, state exactly:\n"
            f'   "{self.abstention_message}"\n'
            "5. Never invent or hallucinate document IDs or information.\n\n"
            f"=== RETRIEVED CONTEXT PASSAGES ===\n{context_str}\n\n"
            f"=== USER QUESTION ===\n{query.strip()}\n\n"
            "=== GROUNDED ANSWER ==="
        )
        return prompt

    def generate(
        self,
        query: str,
        context_chunks: list[RetrievalResult],
    ) -> GenerationResult:
        """
        Execute grounded generation over relevant context chunks.
        
        Args:
            query: User's question.
            context_chunks: Relevant chunks that passed grading.
            
        Returns:
            GenerationResult containing consolidated answer, citations, and metadata.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return GenerationResult(
                query=query,
                answer="Please provide a valid question.",
                cited_doc_ids=[],
                context_used="",
                has_sufficient_context=False,
                raw_prompt="",
            )

        if not context_chunks:
            logger.info("No context chunks provided to GroundedGenerator. Abstaining.")
            return GenerationResult(
                query=clean_query,
                answer=self.abstention_message,
                cited_doc_ids=[],
                context_used="",
                has_sufficient_context=False,
                raw_prompt="",
            )

        context_str = self.build_context_string(context_chunks)
        prompt = self.build_prompt(clean_query, context_str)
        valid_doc_ids = {c.doc_id for c in context_chunks}

        try:
            raw_answer = self.llm_provider.generate(prompt)
            clean_answer = raw_answer.strip()
        except ProviderError as pe:
            logger.error(f"LLM generation failed: {pe}")
            return GenerationResult(
                query=clean_query,
                answer=f"Generation failed due to provider error: {str(pe)}",
                cited_doc_ids=[],
                context_used=context_str,
                has_sufficient_context=False,
                raw_prompt=prompt,
            )
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return GenerationResult(
                query=clean_query,
                answer=f"Generation failed due to unexpected error: {str(e)}",
                cited_doc_ids=[],
                context_used=context_str,
                has_sufficient_context=False,
                raw_prompt=prompt,
            )

        # Detect abstention in answer
        is_abstention = (
            self.abstention_message.lower() in clean_answer.lower()
            or "cannot answer" in clean_answer.lower()
            or "not enough information" in clean_answer.lower()
            or "context does not contain" in clean_answer.lower()
            or "not present in the retrieved context" in clean_answer.lower()
        )

        cited_docs = self.extract_citations(clean_answer, valid_doc_ids)

        return GenerationResult(
            query=clean_query,
            answer=clean_answer,
            cited_doc_ids=cited_docs,
            context_used=context_str,
            has_sufficient_context=not is_abstention,
            raw_prompt=prompt,
        )


# ------------------------------------------------------------------------------
# Functional Interface
# ------------------------------------------------------------------------------

def generate_grounded_answer(
    query: str,
    context_chunks: list[RetrievalResult],
    llm_provider: BaseLLMProvider,
    abstention_message: str = DEFAULT_ABSTENTION_MESSAGE,
) -> GenerationResult:
    """
    Public functional interface for grounded response generation.
    """
    generator = GroundedGenerator(llm_provider=llm_provider, abstention_message=abstention_message)
    return generator.generate(query=query, context_chunks=context_chunks)
