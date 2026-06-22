from __future__ import annotations

import os
from html import escape
from typing import Any
from urllib.parse import urlparse

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
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str):
            return message
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
    if not sources:
        return

    st.markdown('<p class="sources-title">Sources</p>', unsafe_allow_html=True)
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
    st.session_state.setdefault("index_status", None)
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("answer_sources", [])
    st.session_state.setdefault("query_status", None)
    st.session_state.setdefault("recent_repos", [])


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:opsz,wght@9..40,400;9..40,500&display=swap');

        :root {
            --cream: #FDF8F4;
            --surface: #FFFDFB;
            --peach: #F0D9C8;
            --peach-soft: #F8EDE4;
            --orange: #E87D30;
            --input-orange: #F3A463;
            --input-orange-border: #E88B3E;
            --brown: #7A4F2E;
            --brown-soft: #A07858;
        }

        html, body, [class*="css"] {
            font-family: "DM Sans", sans-serif;
            color: var(--brown);
        }

        .stApp {
            background: var(--cream);
        }

        header, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"],
        [data-testid="stDeployButton"] {
            display: none;
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0;
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 0.9rem;
        }

        [data-testid="stSidebar"] {
            background: #FFF7F0;
            border-right: 1px solid var(--peach);
            min-width: 420px !important;
            width: 420px !important;
        }

        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            min-width: 420px !important;
            width: 420px !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 2.4rem 1.6rem;
            width: 420px !important;
        }

        [data-testid="stSidebar"] h2 {
            color: var(--orange);
            font-size: 0.92rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0 0 0.8rem;
        }

        .recent-empty {
            color: #C09A75;
            font-size: 1.05rem;
            font-style: italic;
            line-height: 1.45;
        }

        .recent-meta {
            color: var(--brown-soft);
            font-size: 0.98rem;
            line-height: 1.35;
            margin: -0.5rem 0 0.65rem;
        }

        [data-testid="stSidebar"] .stButton > button {
            min-height: 3.55rem;
            justify-content: flex-start;
            border-color: var(--peach);
            background: #FFFDFB;
            color: var(--brown);
            font-size: 1.05rem;
            line-height: 1.2;
            padding: 0.75rem 1rem;
        }

        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button:focus {
            border-color: var(--orange);
            background: #FFF4E8;
            color: var(--brown);
        }

        .block-container {
            max-width: 1380px;
            padding: 0.9rem 4.5rem 3.4rem;
        }

        h1 {
            color: #231815;
            font-size: clamp(3.6rem, 6vw, 5.4rem);
            line-height: 0.98;
            letter-spacing: 0;
            font-weight: 500;
            padding-top: 0;
            margin-top: 0;
            margin-bottom: 0.45rem;
        }

        [data-testid="stMarkdownContainer"] h1 {
            padding-top: 0;
            margin-top: 0;
        }

        .lede {
            color: var(--brown-soft);
            font-size: 1.38rem;
            line-height: 1.35;
            margin: 0 0 2.8rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--peach);
            border-radius: 8px;
            padding: 2.1rem 3rem;
            box-shadow: none;
        }

        [data-testid="stVerticalBlockBorderWrapper"] + [data-testid="stVerticalBlockBorderWrapper"] {
            margin-top: 1.25rem;
        }

        .step-label,
        .answer-label,
        .sources-title {
            color: var(--orange);
            font-size: 1.08rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 500;
            margin: 0 0 1rem;
        }

        .sources-title {
            color: var(--brown-soft);
            font-size: 0.92rem;
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
        }

        label, [data-testid="stWidgetLabel"] p {
            color: var(--brown-soft);
            font-size: 1.1rem;
            font-weight: 500;
        }

        input, textarea {
            background: var(--input-orange) !important;
            border: 1px solid var(--input-orange-border) !important;
            border-radius: 8px !important;
            color: #FFFDFB !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
            caret-color: #FFFDFB !important;
        }

        [data-testid="stTextInput"] > div,
        [data-testid="stTextInput"] > div > div {
            height: 4.6rem !important;
            min-height: 4.6rem !important;
        }

        [data-testid="stTextInput"] input {
            font-family: "DM Mono", monospace;
            font-size: 1rem;
            height: 4.6rem !important;
            min-height: 4.6rem !important;
            line-height: 1.2 !important;
            padding: 1.05rem 1.1rem !important;
        }

        textarea {
            font-family: "DM Sans", sans-serif !important;
            font-size: 1.35rem !important;
            line-height: 1.35 !important;
            min-height: 10rem !important;
            padding: 1.05rem 1.1rem !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: rgba(255, 253, 251, 0.88) !important;
            opacity: 1 !important;
        }

        .stButton > button {
            min-height: 4.4rem;
            border-radius: 8px;
            border: 1px solid var(--peach);
            background: var(--surface);
            color: var(--orange);
            font-weight: 500;
            font-size: 1.1rem;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stButton > button:focus {
            border-color: var(--orange);
            color: var(--orange);
            background: #FFF8F1;
        }

        .stButton > button:disabled {
            color: #D8C8BC;
            border-color: #EFE6DF;
            background: #FFFDFB;
        }

        .stSlider,
        .stSlider > div,
        .stSlider [data-baseweb="slider"] {
            background: transparent !important;
        }

        .stSlider [data-baseweb="slider"] > div,
        .stSlider [data-baseweb="slider"] > div > div {
            background-color: transparent !important;
            box-shadow: none !important;
        }

        .stSlider [data-baseweb="slider"] > div > div > div {
            background-color: #EAD9CD !important;
            height: 0.28rem;
        }

        .stSlider [role="slider"] {
            background-color: var(--orange);
            border-color: var(--orange);
            box-shadow: 0 0 0 4px #FFFDFB, 0 0 0 6px var(--orange);
        }

        .stSlider [data-testid="stTickBar"] {
            display: none;
        }

        .callout {
            border: 1px solid #F4C99F;
            border-radius: 8px;
            background: #FFF4E8;
            color: #A85E18;
            padding: 0.9rem 1.1rem;
            font-weight: 500;
            margin-top: 1rem;
            font-size: 1.06rem;
        }

        .callout.success {
            color: var(--brown);
        }

        .answer-rule {
            border-top: 1px solid var(--peach);
            margin: 1.5rem 0 1.15rem;
        }

        .empty-answer {
            color: #C09A75;
            font-style: italic;
            font-size: 1.26rem;
            line-height: 1.55;
        }

        .answer-body {
            color: var(--brown);
            font-size: 1.12rem;
            line-height: 1.65;
        }

        code {
            color: var(--brown) !important;
            background: #F8EDE4 !important;
            border: 1px solid #F0D9C8;
            border-radius: 6px;
            font-family: "DM Mono", monospace !important;
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 2rem 1rem 2.8rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 1.8rem 1.25rem;
            }

            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            [data-testid="stTextInput"] input {
                font-size: 0.95rem;
                height: 4rem;
                min-height: 4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_callout(message: str, variant: str = "neutral") -> None:
    st.markdown(
        f'<div class="callout {variant}">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def repo_display_name(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if path:
        return path.removesuffix(".git")
    return repo_url


def remember_recent_repo(repo_url: str, result: dict[str, Any]) -> None:
    repo_id = result.get("repo_id")
    if not repo_id:
        return

    repo = {
        "repo_id": repo_id,
        "repo_url": repo_url,
        "name": repo_display_name(repo_url),
        "files_scanned": result.get("files_scanned", 0),
        "chunks_created": result.get("chunks_created", 0),
    }
    previous = [
        item
        for item in st.session_state.recent_repos
        if item.get("repo_id") != repo_id
    ]
    st.session_state.recent_repos = [repo, *previous][:6]


def render_recent_repos_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Recent Repos")
        if not st.session_state.recent_repos:
            st.markdown(
                '<p class="recent-empty">Indexed repositories will appear here.</p>',
                unsafe_allow_html=True,
            )
            return

        for index, repo in enumerate(st.session_state.recent_repos):
            label = repo.get("name") or repo.get("repo_url") or "Repository"
            if st.button(label, key=f"recent_repo_{index}", use_container_width=True):
                st.session_state.repo_id = repo.get("repo_id")
                st.session_state.repo_url = repo.get("repo_url", "")
                st.session_state.index_status = f"Using {label}."
                st.session_state.query_status = None
                st.session_state.answer = None
                st.session_state.answer_sources = []
                st.rerun()
            st.markdown(
                '<p class="recent-meta">'
                f'{escape(str(repo.get("files_scanned", 0)))} files / '
                f'{escape(str(repo.get("chunks_created", 0)))} chunks'
                "</p>",
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="AI Codebase Engineer",
        page_icon="</>",
        layout="wide",
    )
    initialize_state()
    apply_theme()
    render_recent_repos_sidebar()

    st.markdown("# AI Codebase<br>Engineer", unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">Index a GitHub repository, then ask questions about its code.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<p class="step-label">Step 01 - Index</p>', unsafe_allow_html=True)
        with st.form("index_repository_form"):
            url_column, button_column = st.columns([4.4, 1.25])
            with url_column:
                repo_url = st.text_input(
                    "GitHub repository URL",
                    value=st.session_state.repo_url,
                    placeholder="https://github.com/owner/repository",
                    label_visibility="collapsed",
                )
            with button_column:
                index_submitted = st.form_submit_button(
                    "Index repository",
                    use_container_width=True,
                )

        if st.session_state.index_status:
            render_callout(st.session_state.index_status, "success")
        elif st.session_state.repo_id:
            render_callout("Repository indexed. You can ask questions now.", "success")
        else:
            render_callout("Index a repository before asking questions.")

    if index_submitted:
        if not repo_url.strip():
            st.session_state.index_status = "Enter a GitHub repository URL."
            st.rerun()
        else:
            with st.spinner("Cloning, scanning, chunking, and indexing repository..."):
                try:
                    result = post_json(
                        "/repos/index",
                        {"repo_url": repo_url.strip()},
                    )
                except BackendError as error:
                    st.session_state.index_status = f"Indexing failed: {error}"
                else:
                    st.session_state.repo_id = result.get("repo_id")
                    st.session_state.repo_url = repo_url.strip()
                    st.session_state.index_summary = result
                    remember_recent_repo(repo_url.strip(), result)
                    files = result.get("files_scanned", 0)
                    chunks = result.get("chunks_created", 0)
                    st.session_state.index_status = (
                        f"Repository indexed: {files} files scanned, "
                        f"{chunks} chunks created."
                    )
                st.rerun()

    with st.container(border=True):
        st.markdown('<p class="step-label">Step 02 - Ask</p>', unsafe_allow_html=True)
        with st.form("query_repository_form"):
            question = st.text_area(
                "Question",
                placeholder=(
                    "e.g. How does authentication work? "
                    "Where is rate limiting handled?"
                ),
                disabled=not st.session_state.repo_id,
            )
            slider_column, value_column, button_column = st.columns([4.3, 0.45, 1])
            with slider_column:
                top_k = st.slider(
                    "Relevant code chunks",
                    min_value=1,
                    max_value=20,
                    value=8,
                    disabled=not st.session_state.repo_id,
                )
            with value_column:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(str(top_k))
            with button_column:
                st.markdown("<br>", unsafe_allow_html=True)
                query_submitted = st.form_submit_button(
                    "Ask",
                    use_container_width=True,
                    disabled=not st.session_state.repo_id,
                )

        st.markdown('<div class="answer-rule"></div>', unsafe_allow_html=True)
        st.markdown('<p class="answer-label">Answer</p>', unsafe_allow_html=True)

        if st.session_state.query_status:
            render_callout(st.session_state.query_status)
        elif st.session_state.answer:
            st.markdown('<div class="answer-body">', unsafe_allow_html=True)
            st.markdown(st.session_state.answer)
            render_sources(st.session_state.answer_sources)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<p class="empty-answer">Your answer will appear here once you ask '
                "a question about the indexed repository.</p>",
                unsafe_allow_html=True,
            )

    if query_submitted:
        if not question.strip():
            st.session_state.query_status = (
                "Enter a question about the indexed repository."
            )
            st.rerun()
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
                    st.session_state.query_status = str(error)
                else:
                    st.session_state.query_status = None
                    st.session_state.answer = result.get(
                        "answer",
                        "No answer was returned.",
                    )
                    st.session_state.answer_sources = result.get("sources", [])
                st.rerun()


if __name__ == "__main__":
    main()
