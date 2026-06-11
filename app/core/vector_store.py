from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import chromadb

from app.core.chunker import CodeChunk


@dataclass(frozen=True)
class SearchResult:
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


class VectorStore:
    DEFAULT_COLLECTION_NAME = "code_chunks"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        chroma_path = Path(
            path or os.getenv("CHROMA_PATH", "./data/indexes/chroma")
        ).expanduser()
        chroma_path.mkdir(parents=True, exist_ok=True)

        self.path = chroma_path.resolve()
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def upsert_chunks(
        self,
        repo_id: str,
        chunks: Iterable[CodeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        chunk_list = list(chunks)
        embedding_list = [list(embedding) for embedding in embeddings]

        if len(chunk_list) != len(embedding_list):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunk_list:
            return
        if any(chunk.repo_id != repo_id for chunk in chunk_list):
            raise ValueError("all chunks must belong to repo_id")
        if any(not embedding for embedding in embedding_list):
            raise ValueError("embeddings must not be empty")

        dimensions = {len(embedding) for embedding in embedding_list}
        if len(dimensions) != 1:
            raise ValueError("all embeddings must have the same dimensions")

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunk_list],
            embeddings=embedding_list,
            documents=[chunk.content for chunk in chunk_list],
            metadatas=[
                {
                    "repo_id": repo_id,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol_name": chunk.symbol_name or "",
                    "symbol_type": chunk.symbol_type or "",
                }
                for chunk in chunk_list
            ],
        )

    def search(
        self,
        repo_id: str,
        query_embedding: Sequence[float],
        k: int = 8,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

        results = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=k,
            where={"repo_id": repo_id},
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        search_results: list[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if document is None or metadata is None:
                continue
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    repo_id=str(metadata["repo_id"]),
                    file_path=str(metadata["file_path"]),
                    language=str(metadata["language"]),
                    start_line=int(metadata["start_line"]),
                    end_line=int(metadata["end_line"]),
                    symbol_name=self._optional_metadata(metadata["symbol_name"]),
                    symbol_type=self._optional_metadata(metadata["symbol_type"]),
                    content=document,
                    distance=float(distance),
                )
            )
        return search_results

    def delete_repo(self, repo_id: str) -> None:
        self.collection.delete(where={"repo_id": repo_id})

    @staticmethod
    def _optional_metadata(value: object) -> str | None:
        return str(value) if value not in {None, ""} else None
