from app.core.cache import SQLiteCache
from app.core.rag_engine import RagEngine
from app.core.retriever import RetrievedChunk


def make_chunk(
    *,
    chunk_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
    content: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        repo_id="repo-123",
        file_path=file_path,
        language="Python",
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        symbol_type=None,
        content=content,
        distance=0.1,
    )


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(
        self, repo_id: str, query: str, k: int = 8
    ) -> list[RetrievedChunk]:
        self.calls.append((repo_id, query, k))
        return self.chunks


def test_formats_retrieved_context_with_paths_and_line_ranges() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/auth/routes.py",
            start_line=10,
            end_line=45,
            content="def login_user(...):\n    ...\n",
        ),
        make_chunk(
            chunk_id="chunk-2",
            file_path="app/models/user.py",
            start_line=1,
            end_line=40,
            content="class User(Base):\n    ...\n",
        ),
    ]
    engine = RagEngine(retriever=StubRetriever(chunks), api_key="")

    context = engine.format_context(chunks)
    prompt = engine.build_prompt("How does login work?", chunks)

    assert context == (
        "[1] app/auth/routes.py:10-45\n"
        "def login_user(...):\n"
        "    ...\n\n"
        "[2] app/models/user.py:1-40\n"
        "class User(Base):\n"
        "    ..."
    )
    assert "using only the provided code context" in prompt
    assert "Question:\nHow does login work?" in prompt
    assert f"Code context:\n{context}" in prompt


def test_answer_passes_formatted_prompt_to_llm_and_forces_citation() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/auth/routes.py",
            start_line=10,
            end_line=45,
            content="def login_user(...):\n    ...\n",
        )
    ]
    retriever = StubRetriever(chunks)
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Login is handled by login_user."

    engine = RagEngine(retriever=retriever, llm=llm)

    answer = engine.answer("repo-123", "How does login work?", k=3)

    assert retriever.calls == [("repo-123", "How does login work?", 3)]
    assert "[1] app/auth/routes.py:10-45" in prompts[0]
    assert answer.endswith("Source: `app/auth/routes.py:10-45`")


def test_mock_llm_mode_returns_a_cited_answer() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/models/user.py",
            start_line=1,
            end_line=40,
            content="class User(Base):\n    ...\n",
        )
    ]
    engine = RagEngine(retriever=StubRetriever(chunks), api_key="")

    answer = engine.answer("repo-123", "Where is the user model?")

    assert engine.is_mock
    assert "app/models/user.py:1-40" in answer


def test_no_context_reports_uncertainty_without_calling_llm() -> None:
    called = False

    def llm(_: str) -> str:
        nonlocal called
        called = True
        return "unused"

    engine = RagEngine(retriever=StubRetriever([]), llm=llm)

    answer = engine.answer("repo-123", "Where is authentication?")

    assert "Additional relevant source files" in answer
    assert not called


def test_final_answers_are_cached_by_repo_and_normalized_question() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/auth/routes.py",
            start_line=10,
            end_line=45,
            content="def login_user(...):\n    ...\n",
        )
    ]
    retriever = StubRetriever(chunks)
    calls = 0

    def llm(_: str) -> str:
        nonlocal calls
        calls += 1
        return "Authentication is handled here."

    engine = RagEngine(
        retriever=retriever,
        llm=llm,
        cache=SQLiteCache(database_url="sqlite:///:memory:"),
    )

    first = engine.answer_with_sources(
        "repo-123",
        "How does   authentication work?",
    )
    second = engine.answer_with_sources(
        "repo-123",
        "  HOW DOES authentication WORK?  ",
    )

    assert second == first
    assert calls == 1
    assert retriever.calls == [
        ("repo-123", "How does   authentication work?", 8)
    ]
