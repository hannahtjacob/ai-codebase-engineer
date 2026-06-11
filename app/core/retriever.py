from __future__ import annotations

from dataclasses import dataclass

from app.core.embedding_service import EmbeddingService
from app.core.vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    repo_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    symbol_type: str | None
    content: str
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        repo_id: str,
        query: str,
        k: int = 8,
    ) -> list[RetrievedChunk]:
        if not repo_id.strip():
            raise ValueError("repo_id must not be empty")
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        [query_embedding] = self.embedding_service.embed_texts([query])
        results = self.vector_store.search(repo_id, query_embedding, k=k)
        return [self._to_retrieved_chunk(result) for result in results]

    @staticmethod
    def _to_retrieved_chunk(result: SearchResult) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=result.chunk_id,
            repo_id=result.repo_id,
            file_path=result.file_path,
            language=result.language,
            start_line=result.start_line,
            end_line=result.end_line,
            symbol_name=result.symbol_name,
            symbol_type=result.symbol_type,
            content=result.content,
            distance=result.distance,
        )
