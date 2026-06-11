from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 300


class BackendError(RuntimeError):
    pass


def api_base_url() -> str:
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{api_base_url()}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError as error:
        raise BackendError(
            "Could not connect to the FastAPI backend. Start it with "
            "`uvicorn app.main:app --reload`."
        ) from error
    except requests.Timeout as error:
        raise BackendError("The backend request timed out.") from error
    except requests.RequestException as error:
        raise BackendError(f"Backend request failed: {error}") from error

    if not response.ok:
        raise BackendError(extract_error_message(response))

    try:
        data = response.json()
    except requests.JSONDecodeError as error:
        raise BackendError("The backend returned an invalid JSON response.") from error

    if not isinstance(data, dict):
        raise BackendError("The backend returned an unexpected response.")
    return data


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except requests.JSONDecodeError:
        return f"Backend error ({response.status_code}): {response.text}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = [
            item.get("msg", "Invalid request")
            for item in detail
            if isinstance(item, dict)
        ]
        if messages:
            return "; ".join(messages)
    return f"Backend request failed with status {response.status_code}."


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Sources")
    if not sources:
        st.info("No source chunks were returned.")
        return

    for source in sources:
        file_path = source.get("file_path", "unknown")
        start_line = source.get("start_line", "?")
        end_line = source.get("end_line", "?")
        symbol_name = source.get("symbol_name")
        citation = f"{file_path}:{start_line}-{end_line}"
        if symbol_name:
            st.markdown(f"- `{citation}` - `{symbol_name}`")
        else:
            st.markdown(f"- `{citation}`")


def initialize_state() -> None:
    st.session_state.setdefault("repo_id", None)
    st.session_state.setdefault("repo_url", "")
    st.session_state.setdefault("index_summary", None)


def main() -> None:
    st.set_page_config(
        page_title="AI Codebase Engineer",
        page_icon="</>",
        layout="wide",
    )
    initialize_state()

    st.title("AI Codebase Engineer")
    st.caption("Index a GitHub repository, then ask questions about its code.")

    with st.sidebar:
        st.header("Backend")
        st.code(api_base_url(), language=None)
        st.caption("Override with the `API_BASE_URL` environment variable.")

    st.header("1. Index Repository")
    with st.form("index_repository_form"):
        repo_url = st.text_input(
            "GitHub repository URL",
            value=st.session_state.repo_url,
            placeholder="https://github.com/owner/repository",
        )
        index_submitted = st.form_submit_button(
            "Index Repository",
            type="primary",
            use_container_width=True,
        )

    if index_submitted:
        if not repo_url.strip():
            st.error("Enter a GitHub repository URL.")
        else:
            with st.spinner("Cloning, scanning, chunking, and indexing repository..."):
                try:
                    result = post_json(
                        "/repos/index",
                        {"repo_url": repo_url.strip()},
                    )
                except BackendError as error:
                    st.error(str(error))
                else:
                    st.session_state.repo_id = result.get("repo_id")
                    st.session_state.repo_url = repo_url.strip()
                    st.session_state.index_summary = result
                    st.success("Repository indexed successfully.")

    if st.session_state.repo_id:
        summary = st.session_state.index_summary or {}
        st.markdown(f"**Repository ID:** `{st.session_state.repo_id}`")
        if summary:
            files_column, chunks_column, time_column = st.columns(3)
            files_column.metric("Files scanned", summary.get("files_scanned", 0))
            chunks_column.metric("Chunks created", summary.get("chunks_created", 0))
            time_column.metric(
                "Indexing time",
                f"{summary.get('indexing_time_seconds', 0):.2f}s",
            )
    else:
        st.info("Index a repository before asking questions.")

    st.divider()
    st.header("2. Ask a Question")
    with st.form("query_repository_form"):
        question = st.text_area(
            "Question",
            placeholder="How does authentication work?",
            disabled=not st.session_state.repo_id,
        )
        top_k = st.slider(
            "Relevant code chunks",
            min_value=1,
            max_value=20,
            value=8,
            disabled=not st.session_state.repo_id,
        )
        query_submitted = st.form_submit_button(
            "Ask Question",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.repo_id,
        )

    if query_submitted:
        if not question.strip():
            st.error("Enter a question about the indexed repository.")
        else:
            with st.spinner("Searching the codebase and generating an answer..."):
                try:
                    result = post_json(
                        "/query",
                        {
                            "repo_id": st.session_state.repo_id,
                            "question": question.strip(),
                            "top_k": top_k,
                        },
                    )
                except BackendError as error:
                    st.error(str(error))
                else:
                    st.subheader("Answer")
                    st.markdown(result.get("answer", "No answer was returned."))
                    render_sources(result.get("sources", []))


if __name__ == "__main__":
    main()
