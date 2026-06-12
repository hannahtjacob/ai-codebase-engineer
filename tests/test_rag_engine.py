from types import SimpleNamespace

import pytest
import requests

from app.core.cache import SQLiteCache
from app.core.rag_engine import (
    MissingOpenAIAPIKeyError,
    OllamaUnavailableError,
    RagEngine,
)
from app.core.retriever import RetrievedChunk
import app.core.rag_engine as rag_engine_module


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
    engine = RagEngine(
        retriever=StubRetriever(chunks),
        provider="ollama",
        api_key="",
    )

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

    engine = RagEngine(retriever=retriever, llm=llm, test_mode=True)

    answer = engine.answer("repo-123", "How does login work?", k=3)

    assert retriever.calls == [("repo-123", "How does login work?", 3)]
    assert "[1] app/auth/routes.py:10-45" in prompts[0]
    assert answer.endswith("Source: `app/auth/routes.py:10-45`")


def test_missing_api_key_raises_clear_error() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/models/user.py",
            start_line=1,
            end_line=40,
            content="class User(Base):\n    ...\n",
        )
    ]
    engine = RagEngine(
        retriever=StubRetriever(chunks),
        provider="openai",
        api_key="",
    )

    with pytest.raises(MissingOpenAIAPIKeyError, match="OPENAI_API_KEY"):
        engine.answer("repo-123", "Where is the user model?")


def test_calls_chat_completions_with_requested_model_and_temperature() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/models/user.py",
            start_line=1,
            end_line=40,
            content="class User(Base):\n    ...\n",
        )
    ]
    calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="The user model is defined here."
                    )
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    engine = RagEngine(
        retriever=StubRetriever(chunks),
        provider="openai",
        api_key="",
        client=client,
    )

    answer = engine.answer("repo-123", "Where is the user model?")

    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["temperature"] == 0.2
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {
        "role": "system",
        "content": RagEngine.SYSTEM_MESSAGE,
    }
    assert "[1] app/models/user.py:1-40" in messages[1]["content"]
    assert answer.endswith("Source: `app/models/user.py:1-40`")


def test_production_mode_uses_api_key_and_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/models/user.py",
            start_line=1,
            end_line=40,
            content="class User(Base):\n    ...\n",
        )
    ]
    client_calls: list[str] = []
    completion_calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> SimpleNamespace:
        completion_calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="The user model is here.")
                )
            ]
        )

    def openai_factory(*, api_key: str) -> SimpleNamespace:
        client_calls.append(api_key)
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )

    monkeypatch.setenv("OPENAI_API_KEY", "production-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(rag_engine_module, "OpenAI", openai_factory)
    engine = RagEngine(retriever=StubRetriever(chunks))

    answer = engine.answer("repo-123", "Where is the user model?")

    assert not engine.test_mode
    assert client_calls == ["production-key"]
    assert completion_calls[0]["model"] == "gpt-4o-mini"
    assert answer.endswith("Source: `app/models/user.py:1-40`")


class StubOllamaResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class StubOllamaClient:
    def __init__(
        self,
        *,
        response: StubOllamaResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> StubOllamaResponse:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def test_ollama_uses_same_rag_prompt_and_parses_chat_response() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            file_path="app/models/user.py",
            start_line=1,
            end_line=40,
            content="class User(Base):\n    ...\n",
        )
    ]
    http_client = StubOllamaClient(
        response=StubOllamaResponse(
            {"message": {"content": "The user model is defined here."}}
        )
    )
    engine = RagEngine(
        retriever=StubRetriever(chunks),
        provider="ollama",
        http_client=http_client,
    )

    answer = engine.answer("repo-123", "Where is the user model?")

    call = http_client.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    payload = call["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen2.5-coder:1.5b"
    assert payload["stream"] is False
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"] == RagEngine.SYSTEM_MESSAGE
    assert "[1] app/models/user.py:1-40" in messages[1]["content"]
    assert answer.endswith("Source: `app/models/user.py:1-40`")


def test_ollama_connection_error_has_startup_commands() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        file_path="app/models/user.py",
        start_line=1,
        end_line=40,
        content="class User(Base):\n    ...\n",
    )
    engine = RagEngine(
        retriever=StubRetriever([chunk]),
        provider="ollama",
        http_client=StubOllamaClient(
            error=requests.ConnectionError("connection refused")
        ),
    )

    with pytest.raises(OllamaUnavailableError) as raised:
        engine.answer("repo-123", "Where is the user model?")

    assert "ollama serve" in str(raised.value)
    assert "ollama pull qwen2.5-coder:1.5b" in str(raised.value)


def test_ollama_is_default_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    engine = RagEngine(retriever=StubRetriever([]))

    assert engine.provider == "ollama"


def test_mock_provider_requires_test_mode() -> None:
    with pytest.raises(ValueError, match="only available in test mode"):
        RagEngine(retriever=StubRetriever([]), provider="mock")


def test_answer_question_returns_api_ready_answer_and_sources() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        file_path="app/models/user.py",
        start_line=1,
        end_line=40,
        content="class User(Base):\n    ...\n",
    )
    engine = RagEngine(
        retriever=StubRetriever([chunk]),
        llm=lambda _: "The model is defined here.",
        test_mode=True,
    )

    result = engine.answer_question(
        "repo-123",
        "Where is the user model?",
        top_k=4,
    )

    assert result == {
        "answer": (
            "The model is defined here.\n\n"
            "Source: `app/models/user.py:1-40`"
        ),
        "sources": [
            {
                "file_path": "app/models/user.py",
                "start_line": 1,
                "end_line": 40,
                "symbol_name": None,
            }
        ],
    }


def test_llm_injection_requires_explicit_test_mode() -> None:
    with pytest.raises(ValueError, match="only available in test mode"):
        RagEngine(
            retriever=StubRetriever([]),
            llm=lambda _: "test answer",
        )


def test_no_context_reports_uncertainty_without_calling_llm() -> None:
    called = False

    def llm(_: str) -> str:
        nonlocal called
        called = True
        return "unused"

    engine = RagEngine(
        retriever=StubRetriever([]),
        llm=llm,
        test_mode=True,
    )

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
        test_mode=True,
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
