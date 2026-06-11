from pathlib import Path

from sqlalchemy import inspect, select

from app.core.indexer import RepositoryIndexer
from app.models.db import (
    CodeChunk,
    QueryLog,
    Repository,
    SourceFile,
    get_engine,
    get_session_factory,
    init_db,
)


def test_init_db_creates_all_tables() -> None:
    engine = get_engine("sqlite:///:memory:")

    returned_engine = init_db(engine)

    assert returned_engine is engine
    assert set(inspect(engine).get_table_names()) == {
        "cache_entries",
        "code_chunks",
        "query_logs",
        "repositories",
        "source_files",
    }


def test_indexer_saves_repository_files_and_chunks(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(
        "def greet(name):\n"
        "    return f'Hello, {name}'\n"
        "\n"
        "class Greeter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "client.ts").write_text(
        "export const answer = 42;\n", encoding="utf-8"
    )
    engine = init_db(get_engine("sqlite:///:memory:"))
    session_factory = get_session_factory(engine)

    result = RepositoryIndexer(session_factory).index_path(
        repo_id="repo-123",
        repo_url="https://example.com/repository.git",
        repository_path=tmp_path,
    )

    assert result.file_count == 2
    assert result.chunk_count == 3

    with session_factory() as session:
        repository = session.get(Repository, "repo-123")
        files = session.scalars(
            select(SourceFile).order_by(SourceFile.relative_path)
        ).all()
        chunks = session.scalars(
            select(CodeChunk).order_by(CodeChunk.file_path, CodeChunk.start_line)
        ).all()

        assert repository is not None
        assert repository.url == "https://example.com/repository.git"
        assert repository.local_path == str(tmp_path.resolve())
        assert [(file.relative_path, file.language, file.line_count) for file in files] == [
            ("client.ts", "TypeScript", 1),
            ("service.py", "Python", 5),
        ]
        assert [
            (chunk.file_path, chunk.symbol_name, chunk.symbol_type)
            for chunk in chunks
        ] == [
            ("client.ts", None, None),
            ("service.py", "greet", "function"),
            ("service.py", "Greeter", "class"),
        ]
        assert all(chunk.source_file.repository_id == "repo-123" for chunk in chunks)


def test_reindex_replaces_stale_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "module.py"
    source_path.write_text("def old():\n    pass\n", encoding="utf-8")
    engine = init_db(get_engine("sqlite:///:memory:"))
    session_factory = get_session_factory(engine)
    indexer = RepositoryIndexer(session_factory)

    indexer.index_path(
        repo_id="repo-123",
        repo_url="local",
        repository_path=tmp_path,
    )
    source_path.write_text("def new():\n    return 1\n", encoding="utf-8")
    indexer.index_path(
        repo_id="repo-123",
        repo_url="local",
        repository_path=tmp_path,
    )

    with session_factory() as session:
        files = session.scalars(select(SourceFile)).all()
        chunks = session.scalars(select(CodeChunk)).all()

        assert len(files) == 1
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "new"


def test_query_log_can_be_saved(tmp_path: Path) -> None:
    engine = init_db(get_engine("sqlite:///:memory:"))
    session_factory = get_session_factory(engine)

    with session_factory.begin() as session:
        session.add(
            Repository(
                id="repo-123",
                url="local",
                local_path=str(tmp_path),
            )
        )
        session.add(
            QueryLog(
                repository_id="repo-123",
                query="Where is authentication handled?",
                response="In app/auth.py.",
                duration_ms=12.5,
            )
        )

    with session_factory() as session:
        query_log = session.scalar(select(QueryLog))

        assert query_log is not None
        assert query_log.repository_id == "repo-123"
        assert query_log.duration_ms == 12.5
