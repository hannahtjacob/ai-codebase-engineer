from pathlib import Path

from fastapi.testclient import TestClient

from app.core.indexer import IndexingResult
from app.core.rag_engine import RagResult
from app.core.retriever import RetrievedChunk
from app.main import create_app
from app.models.db import (
    CodeChunk,
    QueryLog,
    Repository,
    SourceFile,
    get_engine,
    get_session_factory,
    init_db,
)


class StubIndexer:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def index_url(self, repo_url: str) -> IndexingResult:
        self.urls.append(repo_url)
        return IndexingResult(
            repo_id="abc123",
            repository_path="/tmp/repository",
            file_count=12,
            chunk_count=34,
        )


class StubRagEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def answer_with_sources(
        self, repo_id: str, question: str, k: int = 8
    ) -> RagResult:
        self.calls.append((repo_id, question, k))
        return RagResult(
            answer="Authentication is handled by login_user "
            "[app/auth/routes.py:10-45].",
            sources=(
                RetrievedChunk(
                    chunk_id="chunk-1",
                    repo_id=repo_id,
                    file_path="app/auth/routes.py",
                    language="Python",
                    start_line=10,
                    end_line=45,
                    symbol_name="login_user",
                    symbol_type="function",
                    content="def login_user():\n    pass\n",
                    distance=0.1,
                ),
            ),
        )


def make_client(
    tmp_path: Path,
) -> tuple[TestClient, StubIndexer, StubRagEngine, object]:
    engine = init_db(get_engine("sqlite:///:memory:"))
    session_factory = get_session_factory(engine)
    indexer = StubIndexer()
    rag_engine = StubRagEngine()
    app = create_app(
        session_factory=session_factory,
        indexer=indexer,  # type: ignore[arg-type]
        rag_engine=rag_engine,  # type: ignore[arg-type]
    )
    return TestClient(app), indexer, rag_engine, session_factory


def seed_repository(session_factory: object, tmp_path: Path) -> None:
    with session_factory.begin() as session:  # type: ignore[attr-defined]
        repository = Repository(
            id="abc123",
            url="https://github.com/some/repo",
            local_path=str(tmp_path / "repository"),
        )
        source_file = SourceFile(
            repository=repository,
            relative_path="app/auth/routes.py",
            absolute_path=str(tmp_path / "repository/app/auth/routes.py"),
            language="Python",
            line_count=45,
        )
        session.add_all(
            [
                repository,
                source_file,
                CodeChunk(
                    chunk_id="chunk-1",
                    repository=repository,
                    source_file=source_file,
                    file_path="app/auth/routes.py",
                    language="Python",
                    start_line=10,
                    end_line=45,
                    symbol_name="login_user",
                    symbol_type="function",
                    content="def login_user():\n    pass\n",
                ),
            ]
        )


def test_post_repos_index_returns_summary(tmp_path: Path) -> None:
    client, indexer, _, _ = make_client(tmp_path)

    with client:
        response = client.post(
            "/repos/index",
            json={"repo_url": "https://github.com/some/repo"},
        )

    assert response.status_code == 201
    assert response.json()["repo_id"] == "abc123"
    assert response.json()["files_scanned"] == 12
    assert response.json()["chunks_created"] == 34
    assert response.json()["indexing_time_seconds"] >= 0
    assert len(indexer.urls) == 1


def test_get_repository_returns_indexing_metadata(tmp_path: Path) -> None:
    client, _, _, session_factory = make_client(tmp_path)
    seed_repository(session_factory, tmp_path)

    with client:
        response = client.get("/repos/abc123")

    assert response.status_code == 200
    assert response.json()["repo_id"] == "abc123"
    assert response.json()["file_count"] == 1
    assert response.json()["chunk_count"] == 1


def test_post_query_returns_sources_and_saves_history(tmp_path: Path) -> None:
    client, _, rag_engine, session_factory = make_client(tmp_path)
    seed_repository(session_factory, tmp_path)

    with client:
        response = client.post(
            "/query",
            json={
                "repo_id": "abc123",
                "question": "How does authentication work?",
                "top_k": 8,
            },
        )
        history_response = client.get("/query/history/abc123")

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Authentication is handled by login_user "
            "[app/auth/routes.py:10-45]."
        ),
        "sources": [
            {
                "file_path": "app/auth/routes.py",
                "start_line": 10,
                "end_line": 45,
                "symbol_name": "login_user",
            }
        ],
    }
    assert rag_engine.calls == [
        ("abc123", "How does authentication work?", 8)
    ]
    assert history_response.status_code == 200
    assert history_response.json()[0]["question"] == (
        "How does authentication work?"
    )

    with session_factory() as session:
        assert session.query(QueryLog).count() == 1


def test_missing_repository_returns_404(tmp_path: Path) -> None:
    client, _, _, _ = make_client(tmp_path)

    with client:
        metadata_response = client.get("/repos/missing")
        query_response = client.post(
            "/query",
            json={"repo_id": "missing", "question": "Where is auth?"},
        )
        history_response = client.get("/query/history/missing")
        graph_response = client.get("/graph/missing")

    assert metadata_response.status_code == 404
    assert query_response.status_code == 404
    assert history_response.status_code == 404
    assert graph_response.status_code == 404


def test_rejects_non_github_repository_url(tmp_path: Path) -> None:
    client, _, _, _ = make_client(tmp_path)

    with client:
        response = client.post(
            "/repos/index",
            json={"repo_url": "https://example.com/repository"},
        )

    assert response.status_code == 422


def test_get_repository_graph_returns_nodes_and_edges(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    (repository_path / "helpers.py").write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (repository_path / "service.py").write_text(
        "from helpers import helper\n"
        "\n"
        "def run():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    client, _, _, session_factory = make_client(tmp_path)
    seed_repository(session_factory, tmp_path)

    with client:
        response = client.get("/graph/abc123")

    assert response.status_code == 200
    payload = response.json()
    assert {
        (node["type"], node["name"])
        for node in payload["nodes"]
    } >= {
        ("Repository", "abc123"),
        ("File", "helpers.py"),
        ("File", "service.py"),
        ("Function", "helper"),
        ("Function", "run"),
        ("Import", "helpers.helper"),
    }
    assert {
        (edge["source"], edge["target"], edge["type"])
        for edge in payload["edges"]
    } >= {
        ("file:service.py", "file:helpers.py", "IMPORTS"),
        (
            "function:service.py:run",
            "function:helpers.py:helper",
            "CALLS",
        ),
    }
