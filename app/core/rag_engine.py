from __future__ import annotations

import os
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.core.cache import SQLiteCache
from app.core.retriever import RetrievedChunk, Retriever


DEFAULT_PROMPT_TEMPLATE = """You are an AI codebase engineer. Answer the user's question using only the provided code context.

Rules:
- Cite file paths and line ranges.
- If the answer is uncertain, say what additional files would be needed.
- Do not invent files, functions, or behavior.
- For modification questions, provide a step-by-step implementation plan.

Question:
{question}

Code context:
{context}
"""


@dataclass(frozen=True)
class RagResult:
    answer: str
    sources: tuple[RetrievedChunk, ...]


class RagEngine:
    DEFAULT_MODEL = "gpt-5-mini"

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        llm: Callable[[str], str] | None = None,
        prompt_template: str | None = None,
        cache: SQLiteCache | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        environment_key = os.getenv("OPENAI_API_KEY")
        resolved_key = api_key if api_key is not None else environment_key
        self.api_key = resolved_key.strip() if resolved_key else None
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", self.DEFAULT_MODEL)
        self.retriever = retriever or Retriever()
        self.prompt_template = prompt_template or self._load_prompt_template()
        self._llm = llm
        self._client = client
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds

        if self._llm is None and self._client is None and self.api_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def is_mock(self) -> bool:
        return self._llm is None and self._client is None

    def answer(
        self,
        repo_id: str,
        question: str,
        k: int = 8,
    ) -> str:
        return self.answer_with_sources(repo_id, question, k=k).answer

    def answer_with_sources(
        self,
        repo_id: str,
        question: str,
        k: int = 8,
    ) -> RagResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        cache_key = self._cache_key(repo_id, question)
        cached = self.cache.get(cache_key) if self.cache is not None else None
        cached_result = self._deserialize_result(cached)
        if cached_result is not None:
            return cached_result

        graph_retrieve = getattr(
            self.retriever,
            "retrieve_with_graph_expansion",
            None,
        )
        if callable(graph_retrieve):
            chunks = graph_retrieve(repo_id, question, top_k=k)
        else:
            chunks = self.retriever.retrieve(repo_id, question, k=k)
        if not chunks:
            result = RagResult(
                answer=(
                    "I cannot answer from the provided code context. Additional "
                    "relevant source files or a broader repository index would be "
                    "needed."
                ),
                sources=(),
            )
            self._cache_result(cache_key, result)
            return result

        prompt = self.build_prompt(question, chunks)
        answer = self._generate(prompt, question, chunks).strip()
        if not answer:
            raise RuntimeError("LLM returned an empty answer")
        result = RagResult(
            answer=self._ensure_citation(answer, chunks),
            sources=tuple(chunks),
        )
        self._cache_result(cache_key, result)
        return result

    def format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "(No relevant code context was retrieved.)"

        return "\n\n".join(
            f"[{index}] {chunk.citation}\n{chunk.content.rstrip()}"
            for index, chunk in enumerate(chunks, start=1)
        )

    def build_prompt(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        return self.prompt_template.format(
            question=question.strip(),
            context=self.format_context(chunks),
        )

    def _generate(
        self,
        prompt: str,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        if self._llm is not None:
            return self._llm(prompt)
        if self._client is not None:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
            )
            return response.output_text
        return self._mock_answer(question, chunks)

    @staticmethod
    def _mock_answer(
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        first = chunks[0]
        return (
            f"Mock answer for {question!r}: the most relevant retrieved code is "
            f"`{first.citation}`."
        )

    @staticmethod
    def _ensure_citation(
        answer: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        if any(chunk.citation in answer for chunk in chunks):
            return answer
        return f"{answer}\n\nSource: `{chunks[0].citation}`"

    @staticmethod
    def _load_prompt_template() -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "answer_question.txt"
        try:
            template = prompt_path.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_PROMPT_TEMPLATE
        return template if template.strip() else DEFAULT_PROMPT_TEMPLATE

    @staticmethod
    def _cache_key(repo_id: str, question: str) -> str:
        normalized_question = " ".join(question.split()).casefold()
        digest = hashlib.sha256(
            f"{repo_id}\0{normalized_question}".encode("utf-8")
        ).hexdigest()
        return f"rag-answer:{digest}"

    def _cache_result(self, key: str, result: RagResult) -> None:
        if self.cache is None:
            return
        self.cache.set(
            key,
            {
                "answer": result.answer,
                "sources": [
                    {
                        "chunk_id": source.chunk_id,
                        "repo_id": source.repo_id,
                        "file_path": source.file_path,
                        "language": source.language,
                        "start_line": source.start_line,
                        "end_line": source.end_line,
                        "symbol_name": source.symbol_name,
                        "symbol_type": source.symbol_type,
                        "content": source.content,
                        "distance": source.distance,
                    }
                    for source in result.sources
                ],
            },
            ttl_seconds=self.cache_ttl_seconds,
        )

    @staticmethod
    def _deserialize_result(value: object) -> RagResult | None:
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            return None
        sources_value = value.get("sources")
        if not isinstance(sources_value, list):
            return None

        try:
            sources = tuple(
                RetrievedChunk(
                    chunk_id=str(source["chunk_id"]),
                    repo_id=str(source["repo_id"]),
                    file_path=str(source["file_path"]),
                    language=str(source["language"]),
                    start_line=int(source["start_line"]),
                    end_line=int(source["end_line"]),
                    symbol_name=source.get("symbol_name"),
                    symbol_type=source.get("symbol_type"),
                    content=str(source["content"]),
                    distance=float(source["distance"]),
                )
                for source in sources_value
                if isinstance(source, dict)
            )
        except (KeyError, TypeError, ValueError):
            return None
        if len(sources) != len(sources_value):
            return None
        return RagResult(answer=value["answer"], sources=sources)
