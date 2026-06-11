from app.core.chunker import CodeChunker
from app.core.file_scanner import SourceFile


def make_source_file(
    content: str, language: str = "Python", path: str = "src/example.py"
) -> SourceFile:
    return SourceFile(
        relative_path=path,
        absolute_path=f"/repo/{path}",
        language=language,
        content=content,
        line_count=len(content.splitlines()),
    )


def test_chunks_top_level_python_functions_and_classes() -> None:
    source = make_source_file(
        "MODULE_VALUE = 1\n"
        "\n"
        "def greet(name):\n"
        "    return f'Hello, {name}'\n"
        "\n"
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'hello'\n"
        "\n"
        "async def fetch():\n"
        "    return 42\n"
    )

    chunks = CodeChunker().chunk_file(source, repo_id="repo-123")

    assert [
        (chunk.symbol_name, chunk.symbol_type, chunk.start_line, chunk.end_line)
        for chunk in chunks
    ] == [
        ("greet", "function", 3, 4),
        ("Greeter", "class", 6, 8),
        ("fetch", "async_function", 10, 11),
    ]
    assert chunks[0].content == (
        "def greet(name):\n    return f'Hello, {name}'\n"
    )
    assert chunks[1].content == (
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'hello'\n"
    )
    assert all(chunk.repo_id == "repo-123" for chunk in chunks)
    assert all(chunk.file_path == "src/example.py" for chunk in chunks)
    assert all(len(chunk.chunk_id) == 64 for chunk in chunks)


def test_python_chunk_includes_decorators() -> None:
    source = make_source_file(
        "@decorator\n"
        "def decorated():\n"
        "    return True\n"
    )

    [chunk] = CodeChunker().chunk_file(source, repo_id="repo-123")

    assert (chunk.start_line, chunk.end_line) == (1, 3)
    assert chunk.content == source.content


def test_fallback_uses_overlapping_eighty_line_windows() -> None:
    lines = [f"line {number}\n" for number in range(1, 151)]
    source = make_source_file(
        "".join(lines), language="TypeScript", path="src/example.ts"
    )

    chunks = CodeChunker().chunk_files([source], repo_id="repo-123")

    assert [
        (chunk.start_line, chunk.end_line, chunk.symbol_name, chunk.symbol_type)
        for chunk in chunks
    ] == [
        (1, 80, None, None),
        (61, 140, None, None),
        (121, 150, None, None),
    ]
    assert chunks[0].content == "".join(lines[:80])
    assert chunks[1].content == "".join(lines[60:140])
    assert chunks[2].content == "".join(lines[120:])


def test_chunk_ids_are_stable_and_distinguish_chunks() -> None:
    source = make_source_file("def first():\n    pass\n\ndef second():\n    pass\n")
    chunker = CodeChunker()

    first_run = chunker.chunk_file(source, repo_id="repo-123")
    second_run = chunker.chunk_file(source, repo_id="repo-123")

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]
    assert first_run[0].chunk_id != first_run[1].chunk_id


def test_invalid_python_uses_fallback_chunking() -> None:
    source = make_source_file("def broken(:\n    pass\n")

    [chunk] = CodeChunker().chunk_file(source, repo_id="repo-123")

    assert (chunk.start_line, chunk.end_line) == (1, 2)
    assert chunk.symbol_name is None
    assert chunk.symbol_type is None
