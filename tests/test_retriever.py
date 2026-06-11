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
