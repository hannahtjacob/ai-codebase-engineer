from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.cache import SQLiteCache
from app.core.embedding_service import EmbeddingService
from app.core.indexer import RepositoryIndexer
from app.core.repo_loader import RepoLoader
from app.core.vector_store import VectorStore
from app.models.db import get_engine, get_session_factory, init_db


def format_summary(
    *,
    repo_id: str,
    file_count: int,
    chunk_count: int,
    elapsed_seconds: float,
) -> str:
    return "\n".join(
        (
            "Indexing complete",
            f"repo_id: {repo_id}",
            f"files_scanned: {file_count}",
            f"chunks_created: {chunk_count}",
            f"indexing_time: {elapsed_seconds:.2f}s",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a GitHub repository, generate code embeddings, and persist "
            "its index."
        )
    )
    parser.add_argument("repo_url", help="GitHub repository URL to index")
    args = parser.parse_args()

    started_at = time.perf_counter()
    engine = get_engine()
    init_db(engine)
    cache = SQLiteCache(engine)
    indexer = RepositoryIndexer(
        get_session_factory(engine),
        repo_loader=RepoLoader(os.getenv("REPO_STORAGE_PATH", "data/repos")),
        embedding_service=EmbeddingService(cache=cache),
        vector_store=VectorStore(),
    )
    result = indexer.index_url(args.repo_url)
    elapsed_seconds = time.perf_counter() - started_at

    print(
        format_summary(
            repo_id=result.repo_id,
            file_count=result.file_count,
            chunk_count=result.chunk_count,
            elapsed_seconds=elapsed_seconds,
        )
    )


if __name__ == "__main__":
    main()
