import networkx as nx

from app.core.retriever import RetrievedChunk
from app.core.retriever import Retriever
from app.core.vector_store import SearchResult


class StubEmbeddingService:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return [[0.25, 0.75]]


class StubVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[float], int]] = []

    def search(
        self, repo_id: str, query_embedding: list[float], k: int = 8
    ) -> list[SearchResult]:
        self.calls.append((repo_id, query_embedding, k))
        return [
            SearchResult(
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
            )
        ]


def test_retriever_embeds_query_and_returns_structured_results() -> None:
    embeddings = StubEmbeddingService()
    vector_store = StubVectorStore()
    retriever = Retriever(embeddings, vector_store)

    results = retriever.retrieve("repo-123", "How does login work?", k=4)

    assert embeddings.inputs == [["How does login work?"]]
    assert vector_store.calls == [("repo-123", [0.25, 0.75], 4)]
    assert len(results) == 1
    assert results[0].file_path == "app/auth/routes.py"
    assert results[0].citation == "app/auth/routes.py:10-45"
    assert results[0].symbol_name == "login_user"


def make_retrieved_chunk(
    chunk_id: str,
    file_path: str,
    symbol_name: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        repo_id="repo-123",
        file_path=file_path,
        language="Python",
        start_line=1,
        end_line=5,
        symbol_name=symbol_name,
        symbol_type="function",
        content=f"def {symbol_name}():\n    pass\n",
        distance=0.0,
    )


def test_graph_expansion_adds_imports_calls_and_parent_classes() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "file:app/auth/routes.py",
        type="File",
        name="app/auth/routes.py",
        file_path="app/auth/routes.py",
    )
    graph.add_node(
        "function:app/auth/routes.py:login_user",
        type="Function",
        name="login_user",
        file_path="app/auth/routes.py",
    )
    graph.add_node(
        "file:app/models/user.py",
        type="File",
        name="app/models/user.py",
        file_path="app/models/user.py",
    )
    graph.add_node(
        "function:app/services/tokens.py:create_token",
        type="Function",
        name="create_token",
        file_path="app/services/tokens.py",
    )
    graph.add_node(
        "class:app/auth/routes.py:AuthController",
        type="Class",
        name="AuthController",
        file_path="app/auth/routes.py",
    )
    graph.add_node(
        "function:app/auth/routes.py:AuthController.login_user",
        type="Function",
        name="AuthController.login_user",
        file_path="app/auth/routes.py",
    )
    graph.add_edge(
        "file:app/auth/routes.py",
        "file:app/models/user.py",
        type="IMPORTS",
    )
    graph.add_edge(
        "function:app/auth/routes.py:login_user",
        "function:app/services/tokens.py:create_token",
        type="CALLS",
    )
    graph.add_edge(
        "class:app/auth/routes.py:AuthController",
        "function:app/auth/routes.py:AuthController.login_user",
        type="CONTAINS",
    )

    seed = SearchResult(
        chunk_id="seed",
        repo_id="repo-123",
        file_path="app/auth/routes.py",
        language="Python",
        start_line=10,
        end_line=45,
        symbol_name="login_user",
        symbol_type="function",
        content="def login_user():\n    pass\n",
        distance=0.1,
    )
    vector_store = StubVectorStore()
    vector_store.search = lambda *_args, **_kwargs: [seed]  # type: ignore[method-assign]
    loaded_node_ids: list[str] = []

    def load_chunks(
        repo_id: str,
        node_ids: list[str],
        _: nx.MultiDiGraph,
    ) -> list[RetrievedChunk]:
        assert repo_id == "repo-123"
        loaded_node_ids.extend(node_ids)
        return [
            make_retrieved_chunk("seed", "app/auth/routes.py", "login_user"),
            make_retrieved_chunk("imported", "app/models/user.py", "User"),
            make_retrieved_chunk(
                "called", "app/services/tokens.py", "create_token"
            ),
        ]

    retriever = Retriever(
        StubEmbeddingService(),
        vector_store,
        graph_provider=lambda _: graph,
        chunk_loader=load_chunks,
    )

    results = retriever.retrieve_with_graph_expansion(
        "repo-123",
        "How does login work?",
    )

    assert loaded_node_ids == [
        "file:app/models/user.py",
        "function:app/services/tokens.py:create_token",
        "class:app/auth/routes.py:AuthController",
    ]
    assert [result.chunk_id for result in results] == [
        "seed",
        "imported",
        "called",
    ]
    assert results[0].distance == 0.1
    assert results[1].distance > results[0].distance


def test_graph_expansion_finds_parent_class_for_method_seed() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "file:service.py",
        type="File",
        name="service.py",
        file_path="service.py",
    )
    graph.add_node(
        "class:service.py:Service",
        type="Class",
        name="Service",
        file_path="service.py",
    )
    graph.add_node(
        "function:service.py:Service.run",
        type="Function",
        name="Service.run",
        file_path="service.py",
    )
    graph.add_edge(
        "class:service.py:Service",
        "function:service.py:Service.run",
        type="CONTAINS",
    )
    seed = SearchResult(
        chunk_id="method",
        repo_id="repo-123",
        file_path="service.py",
        language="Python",
        start_line=2,
        end_line=4,
        symbol_name="run",
        symbol_type="function",
        content="def run(self):\n    pass\n",
        distance=0.2,
    )
    vector_store = StubVectorStore()
    vector_store.search = lambda *_args, **_kwargs: [seed]  # type: ignore[method-assign]
    loaded: list[str] = []

    def load_chunks(
        _repo_id: str,
        node_ids: list[str],
        _graph: nx.MultiDiGraph,
    ) -> list[RetrievedChunk]:
        loaded.extend(node_ids)
        return [make_retrieved_chunk("class", "service.py", "Service")]

    retriever = Retriever(
        StubEmbeddingService(),
        vector_store,
        graph_provider=lambda _: graph,
        chunk_loader=load_chunks,
    )

    results = retriever.retrieve_with_graph_expansion(
        "repo-123",
        "What owns run?",
    )

    assert loaded == ["class:service.py:Service"]
    assert [result.chunk_id for result in results] == ["method", "class"]


def test_zero_expansion_depth_returns_vector_results_only() -> None:
    retriever = Retriever(
        StubEmbeddingService(),
        StubVectorStore(),
        graph_provider=lambda _: nx.MultiDiGraph(),
        chunk_loader=lambda *_: [],
    )

    results = retriever.retrieve_with_graph_expansion(
        "repo-123",
        "How does login work?",
        expansion_depth=0,
    )

    assert [result.chunk_id for result in results] == ["chunk-1"]
