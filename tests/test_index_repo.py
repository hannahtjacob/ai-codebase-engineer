from pathlib import Path

from sqlalchemy import func, select

from app.core.embedding_service import EmbeddingService
from app.core.indexer import RepositoryIndexer
from app.core.vector_store import VectorStore
from app.models.db import (
    CodeChunk,
    Repository,
    SourceFile,
    get_engine,
    get_session_factory,
    init_db,
)
from scripts.index_repo import format_summary


def test_complete_indexing_flow_saves_sqlite_and_vectors(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    (repository_path / "service.py").write_text(
        "def greet(name):\n"
        "    return f'Hello, {name}'\n"
        "\n"
        "class Greeter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    engine = init_db(get_engine("sqlite:///:memory:"))
    session_factory = get_session_factory(engine)
    vector_store = VectorStore(tmp_path / "chroma")
    indexer = RepositoryIndexer(
        session_factory,
        embedding_service=EmbeddingService(api_key="", fake_dimensions=16),
        vector_store=vector_store,
    )

    result = indexer.index_path(
        repo_id="repo-123",
        repo_url="https://github.com/example/project.git",
        repository_path=repository_path,
    )

    assert result.repo_id == "repo-123"
    assert result.file_count == 1
    assert result.chunk_count == 2

    with session_factory() as session:
        assert session.get(Repository, "repo-123") is not None
        assert session.scalar(select(func.count()).select_from(SourceFile)) == 1
        assert session.scalar(select(func.count()).select_from(CodeChunk)) == 2

    [query_embedding] = indexer.embedding_service.embed_texts(
        ["def greet(name):"]
    )
    results = vector_store.search("repo-123", query_embedding)
    assert {result.symbol_name for result in results} == {"greet", "Greeter"}


def test_reindex_replaces_stale_vectors(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    source_path = repository_path / "service.py"
    source_path.write_text("def old():\n    pass\n", encoding="utf-8")
    engine = init_db(get_engine("sqlite:///:memory:"))
    vector_store = VectorStore(tmp_path / "chroma")
    embedding_service = EmbeddingService(api_key="", fake_dimensions=16)
    indexer = RepositoryIndexer(
        get_session_factory(engine),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexer.index_path(
        repo_id="repo-123",
        repo_url="https://github.com/example/project.git",
        repository_path=repository_path,
    )
    source_path.write_text("def new():\n    return 1\n", encoding="utf-8")
    indexer.index_path(
        repo_id="repo-123",
        repo_url="https://github.com/example/project.git",
        repository_path=repository_path,
    )

    [query_embedding] = embedding_service.embed_texts(["def new():"])
    results = vector_store.search("repo-123", query_embedding)
    assert [result.symbol_name for result in results] == ["new"]


def test_summary_contains_required_metrics() -> None:
    summary = format_summary(
        repo_id="abc123",
        file_count=12,
        chunk_count=34,
        elapsed_seconds=1.234,
    )

    assert summary == (
        "Indexing complete\n"
        "repo_id: abc123\n"
        "files_scanned: 12\n"
        "chunks_created: 34\n"
        "indexing_time: 1.23s"
    )
