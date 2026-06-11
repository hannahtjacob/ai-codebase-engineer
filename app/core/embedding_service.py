from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from app.core.cache import SQLiteCache


class EmbeddingServiceError(RuntimeError):
    """Raised when an embedding batch cannot be generated."""


class EmbeddingService:
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_FAKE_DIMENSIONS = 256

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        fake_dimensions: int = DEFAULT_FAKE_DIMENSIONS,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        cache: SQLiteCache | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if initial_backoff < 0:
            raise ValueError("initial_backoff must be non-negative")
        if fake_dimensions <= 0:
            raise ValueError("fake_dimensions must be greater than zero")

        environment_key = os.getenv("OPENAI_API_KEY")
        resolved_key = api_key if api_key is not None else environment_key
        self.api_key = resolved_key.strip() if resolved_key else None
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.fake_dimensions = fake_dimensions
        self._sleep = sleep
        self._client = client
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds

        if self._client is None and self.api_key:
            # Retry behavior is owned here so backoff is predictable and testable.
            self._client = OpenAI(api_key=self.api_key, max_retries=0)

    @property
    def is_fake(self) -> bool:
        return self._client is None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("each text must be a string")
        if not texts:
            return []

        embeddings: list[list[float] | None] = [None] * len(texts)
        missing_by_key: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached = self.cache.get(cache_key) if self.cache is not None else None
            if self._is_embedding(cached):
                embeddings[index] = [float(value) for value in cached]
            else:
                missing_by_key.setdefault(cache_key, []).append(index)

        missing_keys = list(missing_by_key)
        missing_texts = [texts[missing_by_key[key][0]] for key in missing_keys]
        generated: list[list[float]] = []
        for start in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start : start + self.batch_size]
            if not batch:
                break
            if self.is_fake:
                generated.extend(self._fake_embedding(text) for text in batch)
            else:
                generated.extend(self._embed_openai_batch(batch))

        for cache_key, embedding in zip(missing_keys, generated):
            if self.cache is not None:
                self.cache.set(
                    cache_key,
                    embedding,
                    ttl_seconds=self.cache_ttl_seconds,
                )
            for index in missing_by_key[cache_key]:
                embeddings[index] = embedding

        if any(embedding is None for embedding in embeddings):
            raise EmbeddingServiceError("Failed to generate all requested embeddings")
        return [embedding for embedding in embeddings if embedding is not None]

    def _embed_openai_batch(self, texts: Sequence[str]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=list(texts),
                    encoding_format="float",
                )
                ordered_data = sorted(response.data, key=lambda item: item.index)
                embeddings = [list(item.embedding) for item in ordered_data]
                if len(embeddings) != len(texts):
                    raise EmbeddingServiceError(
                        "OpenAI returned a different number of embeddings than requested"
                    )
                return embeddings
            except Exception as error:
                if not self._is_retryable(error) or attempt == self.max_retries:
                    if isinstance(error, EmbeddingServiceError):
                        raise
                    raise EmbeddingServiceError(
                        f"Failed to embed batch after {attempt + 1} attempt(s)"
                    ) from error

                self._sleep(self.initial_backoff * (2**attempt))

        raise AssertionError("embedding retry loop exited unexpectedly")

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(
            error,
            (APIConnectionError, APITimeoutError, RateLimitError),
        ):
            return True
        return isinstance(error, APIStatusError) and error.status_code >= 500

    def _fake_embedding(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0

        while len(values) < self.fake_dimensions:
            digest = hashlib.sha256(
                counter.to_bytes(4, byteorder="big") + text.encode("utf-8")
            ).digest()
            values.extend((byte - 127.5) / 127.5 for byte in digest)
            counter += 1

        vector = values[: self.fake_dimensions]
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]

    def _cache_key(self, text: str) -> str:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mode = (
            f"fake:{self.fake_dimensions}"
            if self.is_fake
            else f"openai:{self.model}"
        )
        return f"embedding:{mode}:{content_hash}"

    @staticmethod
    def _is_embedding(value: object) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, (int, float)) for item in value)
        )
