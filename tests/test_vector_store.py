from pathlib import Path

import pytest

from app.core.chunker import CodeChunk
from app.core.vector_store import VectorStore


def make_chunk(
    chunk_id: str,
    repo_id: str,
    *,
    content: str,
    file_path: str,
    symbol_name: str | None = None,
    symbol_type: str | None = None,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        repo_id=repo_id,
        file_path=file_path,
        language="Python",
        start_line=1,
        end_line=2,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        content=content,
    )


def test_upsert_and_search_are_filtered_by_repository(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    repo_one_chunks = [
        make_chunk(
            "chunk-a",
            "repo-one",
            content="authentication logic",
            file_path="app/auth.py",
            symbol_name="authenticate",
            symbol_type="function",
        ),
        make_chunk(
            "chunk-b",
            "repo-one",
            content="database logic",
            file_path="app/db.py",
        ),
    ]
    repo_two_chunk = make_chunk(
        "chunk-c",
        "repo-two",
        content="other repository",
        file_path="app/other.py",
    )
    store.upsert_chunks("repo-one", repo_one_chunks, [[1.0, 0.0], [0.0, 1.0]])
    store.upsert_chunks("repo-two", [repo_two_chunk], [[1.0, 0.0]])

    results = store.search("repo-one", [1.0, 0.0], k=8)

    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]
    assert all(result.repo_id == "repo-one" for result in results)
    assert results[0].content == "authentication logic"
    assert results[0].file_path == "app/auth.py"
    assert results[0].language == "Python"
    assert (results[0].start_line, results[0].end_line) == (1, 2)
    assert results[0].symbol_name == "authenticate"
    assert results[0].symbol_type == "function"
    assert results[1].symbol_name is None
    assert results[1].symbol_type is None


def test_upsert_replaces_existing_chunk(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    original = make_chunk(
        "chunk-a",
        "repo-one",
        content="old content",
        file_path="old.py",
    )
    replacement = make_chunk(
        "chunk-a",
        "repo-one",
        content="new content",
        file_path="new.py",
    )

    store.upsert_chunks("repo-one", [original], [[1.0, 0.0]])
    store.upsert_chunks("repo-one", [replacement], [[0.0, 1.0]])

    [result] = store.search("repo-one", [0.0, 1.0])
    assert result.content == "new content"
    assert result.file_path == "new.py"


def test_delete_repo_only_removes_matching_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    store.upsert_chunks(
        "repo-one",
        [
            make_chunk(
                "chunk-a",
                "repo-one",
                content="first",
                file_path="first.py",
            )
        ],
        [[1.0, 0.0]],
    )
    store.upsert_chunks(
        "repo-two",
        [
            make_chunk(
                "chunk-b",
                "repo-two",
                content="second",
                file_path="second.py",
            )
        ],
        [[0.0, 1.0]],
    )

    store.delete_repo("repo-one")

    assert store.search("repo-one", [1.0, 0.0]) == []
    assert [result.chunk_id for result in store.search("repo-two", [0.0, 1.0])] == [
        "chunk-b"
    ]


def test_rejects_mismatched_chunks_and_embeddings(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    chunk = make_chunk(
        "chunk-a",
        "repo-one",
        content="content",
        file_path="file.py",
    )

    with pytest.raises(ValueError, match="same length"):
        store.upsert_chunks("repo-one", [chunk], [])


def test_persists_data_at_chroma_path(tmp_path: Path) -> None:
    chroma_path = tmp_path / "chroma"
    first_store = VectorStore(chroma_path)
    first_store.upsert_chunks(
        "repo-one",
        [
            make_chunk(
                "chunk-a",
                "repo-one",
                content="persistent",
                file_path="file.py",
            )
        ],
        [[1.0, 0.0]],
    )

    second_store = VectorStore(chroma_path)

    assert [result.chunk_id for result in second_store.search(
        "repo-one", [1.0, 0.0]
    )] == ["chunk-a"]
