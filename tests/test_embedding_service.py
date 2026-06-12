from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, RateLimitError

from app.core.cache import SQLiteCache
from app.core.embedding_service import EmbeddingService, EmbeddingServiceError


class FakeEmbeddingsResource:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_client(responses: list[object]) -> SimpleNamespace:
    return SimpleNamespace(embeddings=FakeEmbeddingsResource(responses))


def make_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in enumerate(vectors)
        ]
    )


def test_fake_embeddings_are_deterministic_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = EmbeddingService(provider="fake", fake_dimensions=16)

    first = service.embed_texts(["alpha", "beta", "alpha"])
    second = service.embed_texts(["alpha"])

    assert service.is_fake
    assert first[0] == first[2] == second[0]
    assert first[0] != first[1]
    assert all(len(vector) == 16 for vector in first)
    assert sum(value * value for value in first[0]) == pytest.approx(1.0)


def test_local_is_the_default_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    service = EmbeddingService(local_model=FakeLocalModel())

    assert service.provider == "local"


def test_openai_embeddings_are_batched_and_keep_input_order() -> None:
    client = make_client(
        [
            make_response([[1.0], [2.0]]),
            make_response([[3.0], [4.0]]),
            make_response([[5.0]]),
        ]
    )
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        batch_size=2,
        client=client,
    )

    embeddings = service.embed_texts(["one", "two", "three", "four", "five"])

    assert embeddings == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [call["input"] for call in client.embeddings.calls] == [
        ["one", "two"],
        ["three", "four"],
        ["five"],
    ]
    assert all(
        call["model"] == "text-embedding-3-small"
        for call in client.embeddings.calls
    )
    assert all("encoding_format" not in call for call in client.embeddings.calls)


def test_openai_embedding_smoke_inputs_keep_order() -> None:
    client = make_client([make_response([[0.1, 0.2], [0.3, 0.4]])])
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=client,
    )
    texts = ["hello world", "def add(a, b): return a + b"]

    assert service.embed_texts(texts) == [[0.1, 0.2], [0.3, 0.4]]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": texts,
        }
    ]


def test_retries_transient_errors_with_exponential_backoff() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    client = make_client(
        [
            APIConnectionError(request=request),
            APIConnectionError(request=request),
            make_response([[0.25, 0.75]]),
        ]
    )
    delays: list[float] = []
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=client,
        max_retries=2,
        initial_backoff=0.5,
        sleep=delays.append,
    )

    assert service.embed_texts(["hello"]) == [[0.25, 0.75]]
    assert delays == [0.5, 1.0]
    assert len(client.embeddings.calls) == 3


def test_raises_service_error_after_retries_are_exhausted() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    client = make_client(
        [
            APIConnectionError(request=request),
            APIConnectionError(request=request),
        ]
    )
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=client,
        max_retries=1,
        initial_backoff=0,
        sleep=lambda _: None,
    )

    with pytest.raises(
        EmbeddingServiceError,
        match="OpenAI embedding request failed after 2 attempt",
    ):
        service.embed_texts(["hello"])


def test_insufficient_quota_falls_back_without_retrying_openai() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, request=request)
    error = RateLimitError(
        "quota exceeded",
        response=response,
        body={
            "message": "quota exceeded",
            "code": "insufficient_quota",
        },
    )
    client = make_client([error])
    delays: list[float] = []
    local_model = FakeLocalModel()
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=client,
        local_model=local_model,
        max_retries=3,
        sleep=delays.append,
    )

    assert service.embed_texts(["hello"]) == [[5.0, 1.0]]

    assert len(client.embeddings.calls) == 1
    assert delays == []


def test_empty_input_does_not_call_client() -> None:
    client = make_client([])
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=client,
    )

    assert service.embed_texts([]) == []
    assert client.embeddings.calls == []


def test_rejects_non_string_input() -> None:
    service = EmbeddingService(provider="fake")

    with pytest.raises(TypeError, match="each text"):
        service.embed_texts(["valid", 123])  # type: ignore[list-item]


def test_embeddings_are_cached_by_content_hash() -> None:
    client = make_client([make_response([[1.0, 2.0]])])
    cache = SQLiteCache(database_url="sqlite:///:memory:")
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=client,
        cache=cache,
    )

    first = service.embed_texts(["same content", "same content"])
    second = service.embed_texts(["same content"])

    assert first == [[1.0, 2.0], [1.0, 2.0]]
    assert second == [[1.0, 2.0]]
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == ["same content"]


class FakeLocalModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


def test_local_embeddings_return_non_empty_vector() -> None:
    local_model = FakeLocalModel()
    service = EmbeddingService(provider="local", local_model=local_model)

    embeddings = service.embed_texts(["hello world"])

    assert embeddings == [[11.0, 1.0]]
    assert local_model.calls == [["hello world"]]


def test_empty_strings_are_skipped_without_losing_positions() -> None:
    local_model = FakeLocalModel()
    service = EmbeddingService(provider="local", local_model=local_model)

    embeddings = service.embed_texts(["hello", "", " \n", "world"])

    assert embeddings == [[5.0, 1.0], [], [], [5.0, 1.0]]
    assert local_model.calls == [["hello", "world"]]


def test_openai_quota_falls_back_to_local_embeddings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, request=request)
    quota_error = RateLimitError(
        "quota exceeded",
        response=response,
        body={"message": "quota exceeded", "code": "insufficient_quota"},
    )
    local_model = FakeLocalModel()
    service = EmbeddingService(
        provider="openai",
        api_key="test-key",
        client=make_client([quota_error]),
        local_model=local_model,
    )

    with caplog.at_level("WARNING"):
        embeddings = service.embed_texts(["hello"])

    assert embeddings == [[5.0, 1.0]]
    assert service.provider == "local"
    assert "falling back to local embeddings" in caplog.text
