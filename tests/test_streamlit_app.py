from typing import Any

import pytest
import requests

from frontend.streamlit_app import BackendError, extract_error_message, post_json


class StubResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.ok = 200 <= status_code < 400

    def json(self) -> Any:
        return self._payload


def test_post_json_returns_backend_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> StubResponse:
        calls.append({"url": url, **kwargs})
        return StubResponse(payload={"repo_id": "abc123"})

    monkeypatch.setenv("API_BASE_URL", "http://backend:9000/")
    monkeypatch.setattr(requests, "post", fake_post)

    result = post_json("/repos/index", {"repo_url": "https://github.com/a/b"})

    assert result == {"repo_id": "abc123"}
    assert calls[0]["url"] == "http://backend:9000/repos/index"


def test_post_json_reports_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_post(*_: Any, **__: Any) -> None:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fail_post)

    with pytest.raises(BackendError, match="uvicorn app.main:app --reload"):
        post_json("/query", {})


def test_extract_error_message_handles_fastapi_validation_errors() -> None:
    response = StubResponse(
        status_code=422,
        payload={
            "detail": [
                {"msg": "Field required"},
                {"msg": "Input should be greater than or equal to 1"},
            ]
        },
    )

    assert extract_error_message(response) == (
        "Field required; Input should be greater than or equal to 1"
    )


def test_post_json_preserves_backend_indexing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(*_: Any, **__: Any) -> StubResponse:
        return StubResponse(
            status_code=502,
            payload={
                "detail": (
                    "Unable to generate repository embeddings: "
                    "Incorrect API key provided"
                )
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(BackendError, match="Incorrect API key provided"):
        post_json("/repos/index", {"repo_url": "https://github.com/a/b"})
