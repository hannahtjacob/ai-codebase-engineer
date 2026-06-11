from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import sessionmaker

from app.core.chunker import CodeChunker
from app.core.embedding_service import EmbeddingService
from app.core.file_scanner import FileScanner
from app.core.repo_loader import RepoLoader
from app.core.vector_store import VectorStore
from app.models.db import (
    CodeChunk as CodeChunkRecord,
    Repository,
    SourceFile as SourceFileRecord,
)


@dataclass(frozen=True)
class IndexingResult:
    repo_id: str
    repository_path: str
    file_count: int
    chunk_count: int


class RepositoryIndexer:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        scanner: FileScanner | None = None,
        chunker: CodeChunker | None = None,
        repo_loader: RepoLoader | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        if (embedding_service is None) != (vector_store is None):
            raise ValueError(
                "embedding_service and vector_store must be provided together"
            )

        self.session_factory = session_factory
        self.scanner = scanner or FileScanner()
        self.chunker = chunker or CodeChunker()
        self.repo_loader = repo_loader or RepoLoader()
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_url(self, repo_url: str) -> IndexingResult:
        repo_id = self.repo_loader.repo_id_from_url(repo_url)
        repository_path = self.repo_loader.clone_repo(repo_url)
        return self.index_path(
            repo_id=repo_id,
            repo_url=repo_url,
            repository_path=repository_path,
        )

    def index_path(
        self,
        *,
        repo_id: str,
        repo_url: str,
        repository_path: str | Path,
    ) -> IndexingResult:
        resolved_path = Path(repository_path).expanduser().resolve()
        files = self.scanner.scan(resolved_path)
        chunks = self.chunker.chunk_files(files, repo_id)

        if self.embedding_service is not None and self.vector_store is not None:
            embeddings = self.embedding_service.embed_texts(
                [chunk.content for chunk in chunks]
            )
            self.vector_store.delete_repo(repo_id)
            self.vector_store.upsert_chunks(repo_id, chunks, embeddings)

        with self.session_factory.begin() as session:
            repository = session.get(Repository, repo_id)
            if repository is None:
                repository = Repository(
                    id=repo_id,
                    url=repo_url,
                    local_path=str(resolved_path),
                )
                session.add(repository)
            else:
                repository.url = repo_url
                repository.local_path = str(resolved_path)
                repository.indexed_at = datetime.now(timezone.utc)
                session.execute(
                    delete(CodeChunkRecord).where(
                        CodeChunkRecord.repository_id == repo_id
                    )
                )
                session.execute(
                    delete(SourceFileRecord).where(
                        SourceFileRecord.repository_id == repo_id
                    )
                )

            file_records: dict[str, SourceFileRecord] = {}
            for source_file in files:
                record = SourceFileRecord(
                    repository=repository,
                    relative_path=source_file.relative_path,
                    absolute_path=source_file.absolute_path,
                    language=source_file.language,
                    line_count=source_file.line_count,
                )
                session.add(record)
                file_records[source_file.relative_path] = record

            for chunk in chunks:
                session.add(
                    CodeChunkRecord(
                        chunk_id=chunk.chunk_id,
                        repository=repository,
                        source_file=file_records[chunk.file_path],
                        file_path=chunk.file_path,
                        language=chunk.language,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        symbol_name=chunk.symbol_name,
                        symbol_type=chunk.symbol_type,
                        content=chunk.content,
                    )
                )

        return IndexingResult(
            repo_id=repo_id,
            repository_path=str(resolved_path),
            file_count=len(files),
            chunk_count=len(chunks),
        )
