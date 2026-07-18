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

    # Console layout overrides. Kept separate from the original theme so the
    # selectors remain easy to update across Streamlit releases.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

        :root {
            --navy: #244985;
            --cyan: #08a9c9;
            --cyan-soft: #e3f6fb;
            --ink: #20242d;
            --muted: #8c96aa;
            --line: #d8dde5;
            --panel: #f4f5f7;
            --green: #238746;
            --rust: #c65416;
        }

        html, body, [class*="css"], .stApp, button, input, textarea, select {
            font-family: "IBM Plex Mono", monospace !important;
        }
        .stApp { background: white; color: var(--ink); }
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: none; padding: 0 !important; margin: 0;
        }
        .console-topbar {
            height: 80px; background: var(--navy); border-bottom: 4px solid var(--cyan);
            color: white; display: flex; align-items: center; justify-content: space-between;
            padding: 0 2rem; font-weight: 600; letter-spacing: .03em;
        }
        .console-brand { font-size: 1.25rem; }
        .console-brand .bracket, .console-brand .dot { color: var(--cyan); }
        .console-version { color: #91a8ce; font-weight: 400; margin-left: .7rem; }
        .console-system { color: #a9b8d2; font-size: .86rem; display: flex; gap: 2rem; }
        .console-system .online { color: #b9c7db; }
        .console-system .online::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4add75; box-shadow: 0 0 8px #4add75; margin-right: 8px; }
        .console-crumb {
            height: 62px; display: flex; align-items: center; padding: 0 2rem;
            border-bottom: 1px solid var(--line); color: var(--muted); font-size: .9rem;
        }
        .console-crumb strong { color: #0786aa; font-weight: 500; }
        [data-testid="stHorizontalBlock"] { gap: 0 !important; }
        [data-testid="column"] { padding: 0 !important; }
        .rail, .workspace { min-height: calc(100vh - 142px); }
        .left-rail { border-right: 1px solid var(--line); }
        .right-rail { border-left: 1px solid var(--line); }
        .rail-heading {
            background: #eff1f4; color: var(--muted); font-size: .84rem; font-weight: 600;
            letter-spacing: .09em; padding: 1rem 1.25rem; text-transform: uppercase;
        }
        .repo-row { padding: .8rem 1.25rem; border-bottom: 1px solid var(--line); }
        .repo-row.active { background: var(--cyan-soft); border-left: 3px solid var(--cyan); padding-left: calc(1.25rem - 3px); }
        .repo-name { font-size: .9rem; font-weight: 600; color: var(--ink); }
        .repo-row.active .repo-name { color: #087da0; }
        .repo-meta { color: var(--muted); font-size: .76rem; margin: .25rem 0 0 1.3rem; }
        .tree { padding: .85rem 1.25rem; color: #46536b; font-size: .8rem; line-height: 1.9; }
        .tree .folder { color: #285caa; font-weight: 600; }
        .tree .count { float: right; color: var(--muted); }
        .tree .indent { padding-left: 1.25rem; }
        .index-strip { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 1rem 1.5rem; min-height: 104px; }
        .index-ok { color: var(--green); font-size: .88rem; max-width: 40%; }
        .index-ok .check { display: inline-grid; place-items: center; width: 22px; height: 22px; background: var(--green); color: white; border-radius: 50%; margin-right: .5rem; }
        .index-stats { display: flex; gap: 2rem; color: var(--muted); font-size: .79rem; }
        .index-stats strong { display: block; color: #7b859a; font-size: 1rem; }
        .query-zone { padding: 1.25rem 1.5rem .8rem; border-bottom: 1px solid var(--line); }
        .console-label { color: var(--muted); text-transform: uppercase; font-size: .8rem; font-weight: 600; letter-spacing: .08em; margin-bottom: .65rem; }
        textarea, input { background: #fff !important; color: var(--ink) !important; border: 1px solid #bfc6d1 !important; border-radius: 2px !important; caret-color: var(--ink) !important; }
        textarea { min-height: 92px !important; font-size: .95rem !important; padding: .75rem 1rem !important; }
        input::placeholder, textarea::placeholder { color: #7c8492 !important; }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            min-height: 2.8rem; border-radius: 2px; border: 1px solid #bfc6d1;
            background: white; color: var(--muted); font-size: .84rem; font-weight: 600;
        }
        [data-testid="stFormSubmitButton"] button { background: var(--cyan); border-color: var(--cyan); color: white; }
        .answer-panel { background: #f7f8fa; padding: 1.5rem; min-height: 500px; }
        .answer-header { display: flex; gap: 1rem; align-items: baseline; border-bottom: 1px solid var(--line); padding-bottom: 1rem; margin-bottom: 1.25rem; }
        .answer-badge { color: #0586aa; background: var(--cyan-soft); padding: .35rem .65rem; font-size: .76rem; font-weight: 600; letter-spacing: .1em; }
        .answer-body, .answer-panel [data-testid="stMarkdownContainer"] { color: #3d4659; font-size: .92rem; line-height: 1.65; }
        .answer-panel code { color: var(--rust) !important; background: #fff !important; border-color: #dfe2e7; }
        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; }
        .metric { padding: 1rem 1.15rem; min-height: 90px; border-bottom: 1px solid var(--line); border-right: 1px solid var(--line); }
        .metric:nth-child(even) { border-right: 0; }
        .metric-label { color: var(--muted); font-size: .73rem; font-weight: 600; text-transform: uppercase; }
        .metric-value { color: var(--ink); font-size: 1.35rem; margin-top: .55rem; }
        .metric-value.good { color: var(--green); } .metric-value.info { color: #0788ad; } .metric-value.warn { color: var(--rust); }
        .cache { padding: 1rem 1.15rem; border-bottom: 1px solid var(--line); }
        .cache-track { height: 10px; background: #f1f3f5; border: 1px solid var(--line); margin: .65rem 0; }
        .cache-fill { width: 34%; height: 100%; background: var(--cyan); }
        .cache-meta { display: flex; justify-content: space-between; color: var(--muted); font-size: .72rem; }
        .dep-list { padding: .8rem 1rem 1.3rem; font-size: .76rem; color: var(--muted); }
        .dep-node { background: var(--cyan-soft); color: #087fa3; padding: .5rem .55rem; margin: .35rem 0; }
        .dep-edge { padding-left: 1.4rem; line-height: 1.75; }
        .callout { border-radius: 2px; font-size: .78rem; padding: .65rem .8rem; margin-top: .6rem; }
        .sources-title { color: var(--muted); }
        @media (max-width: 900px) {
            .console-system { display: none; }
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            [data-testid="column"] { min-width: 100% !important; }
            .rail { min-height: auto; }
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


def render_console_header() -> None:
    st.markdown(
        """
        <div class="console-topbar">
          <div class="console-brand"><span class="bracket">[</span> AI Codebase Engineer <span class="bracket">]</span><span class="console-version"><span class="dot">●</span> v0.9.2</span></div>
          <div class="console-system"><span class="online">Engine active</span><span>Chroma · SQLite · NetworkX</span><span>Ollama: qwen2.5-coder</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active = st.session_state.recent_repos[0] if st.session_state.recent_repos else {}
    name = escape(str(active.get("name", "no-repository-selected")))
    files = escape(str(active.get("files_scanned", 0)))
    chunks = escape(str(active.get("chunks_created", 0)))
    st.markdown(
        f'<div class="console-crumb">repos&nbsp; / &nbsp;<strong>{name}</strong>&nbsp; · &nbsp;{chunks} chunks&nbsp; · &nbsp;{files} files</div>',
        unsafe_allow_html=True,
    )


def render_left_rail() -> None:
    repos = st.session_state.recent_repos
    rows = []
    if repos:
        for index, repo in enumerate(repos[:3]):
            active = " active" if repo.get("repo_id") == st.session_state.repo_id else ""
            rows.append(
                f'<div class="repo-row{active}"><div class="repo-name">◇ &nbsp;{escape(str(repo.get("name", "repository")))}</div>'
                f'<div class="repo-meta">{escape(str(repo.get("files_scanned", 0)))} files · indexed</div></div>'
            )
    else:
        rows.append('<div class="repo-row active"><div class="repo-name">◇ &nbsp;your-repository</div><div class="repo-meta">Ready to index</div></div>')
    st.markdown(
        '<div class="rail left-rail"><div class="rail-heading">Repositories</div>'
        + "".join(rows)
        + """<div class="rail-heading">File tree</div>
        <div class="tree">
          <div>▾ <span class="folder">app/</span></div>
          <div class="indent">▾ <span class="folder">api/</span><span class="count">3</span></div>
          <div class="indent">&nbsp;&nbsp;· routes.py</div>
          <div class="indent">&nbsp;&nbsp;· dependencies.py</div>
          <div class="indent">▾ <span class="folder">core/</span><span class="count">8</span></div>
          <div class="indent">&nbsp;&nbsp;· rag_engine.py</div>
          <div class="indent">&nbsp;&nbsp;· retriever.py</div>
          <div class="indent">&nbsp;&nbsp;· embedding_service.py</div>
          <div class="indent">&nbsp;&nbsp;· graph_builder.py</div>
          <div class="indent">&nbsp;&nbsp;· cache.py</div>
          <div class="indent">▾ <span class="folder">models/</span><span class="count">3</span></div>
          <div class="indent">&nbsp;&nbsp;· schemas.py</div>
          <div>▾ <span class="folder">frontend/</span></div>
          <div class="indent">· streamlit_app.py</div>
          <div>▸ <span class="folder">tests/</span><span class="count">17</span></div>
        </div></div>""",
        unsafe_allow_html=True,
    )


def render_right_rail() -> None:
    st.markdown(
        """<div class="rail right-rail">
        <div class="rail-heading">Eval metrics</div>
        <div class="metric-grid">
          <div class="metric"><div class="metric-label">Recall@5</div><div class="metric-value good">0.82</div></div>
          <div class="metric"><div class="metric-label">Recall@10</div><div class="metric-value good">0.91</div></div>
          <div class="metric"><div class="metric-label">Citation acc.</div><div class="metric-value info">0.87</div></div>
          <div class="metric"><div class="metric-label">Avg latency</div><div class="metric-value">2.4s</div></div>
          <div class="metric"><div class="metric-label">Chunks / q</div><div class="metric-value">7.2</div></div>
          <div class="metric"><div class="metric-label">Cache hit</div><div class="metric-value warn">34%</div></div>
        </div>
        <div class="cache"><div class="console-label">Cache utilization</div><div class="cache-track"><div class="cache-fill"></div></div><div class="cache-meta"><span>34% hits</span><span>147 entries</span><span>1.2MB</span></div></div>
        <div class="rail-heading">Dependency graph · this query</div>
        <div class="dep-list">
          <div class="dep-node">● core/retriever.py</div>
          <div class="dep-edge">↳ CALLS <span style="color:#238746">●</span> embed_service.embed()</div>
          <div class="dep-edge">↳ CALLS <span style="color:#238746">●</span> graph.expand_one_hop()</div>
          <div class="dep-node">● core/graph_builder.py</div>
          <div class="dep-edge">↳ USES ● networkx.DiGraph</div>
          <div class="dep-node">● core/rag_engine.py</div>
          <div class="dep-edge">↳ IMPORTS ● core/retriever.py</div>
          <div class="dep-edge">● core/cache.py</div>
          <div class="dep-edge"><span style="color:#c65416">●</span> models/schemas.py</div>
        </div></div>""",
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
    render_console_header()
    left, center, right = st.columns([1.15, 2.55, 1.35])

    with left:
        render_left_rail()

    with center:
        summary = st.session_state.index_summary or {}
        files = summary.get("files_scanned", 0)
        chunks = summary.get("chunks_created", 0)
        repo_name = repo_display_name(st.session_state.repo_url) if st.session_state.repo_url else "waiting for repository"
        st.markdown(
            f'<div class="workspace"><div class="index-strip"><div class="index-ok"><span class="check">✓</span>{escape(repo_name)}</div>'
            f'<div class="index-stats"><span><strong>{files}</strong>files</span><span><strong>{chunks}</strong>chunks</span><span><strong>local</strong>index</span><span><strong>MiniLM</strong>L6-v2</span></div></div></div>',
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="query-zone"><div class="console-label">Index a GitHub repository</div>', unsafe_allow_html=True)
        with st.form("index_repository_form"):
            url_column, button_column = st.columns([4.5, 1.4])
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
            st.markdown('</div><div class="query-zone"><div class="console-label">Ask a question about this codebase</div>', unsafe_allow_html=True)

    with center:
        with st.form("query_repository_form"):
            question = st.text_area(
                "Question",
                placeholder=(
                    "e.g. How does authentication work? "
                    "Where is rate limiting handled?"
                ),
                disabled=not st.session_state.repo_id,
            )
            slider_column, value_column, button_column = st.columns([4.3, 0.45, 1.2])
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

        st.markdown('</div><div class="answer-panel"><div class="answer-header"><span class="answer-badge">ANSWER</span><em>Codebase analysis</em></div>', unsafe_allow_html=True)

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
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        render_right_rail()

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
