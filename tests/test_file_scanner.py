from pathlib import Path

import pytest

from app.core.file_scanner import FileScanner, SourceFile


def test_scan_returns_source_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    return 42\n", encoding="utf-8"
    )
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "app.TS").write_text(
        "export const answer = 42;\n", encoding="utf-8"
    )
    (tmp_path / "notes.txt").write_text("not source code", encoding="utf-8")

    files = FileScanner().scan(tmp_path)

    assert files == [
        SourceFile(
            relative_path="src/main.py",
            absolute_path=str((tmp_path / "src" / "main.py").resolve()),
            language="Python",
            content="def main():\n    return 42\n",
            line_count=2,
        ),
        SourceFile(
            relative_path="web/app.TS",
            absolute_path=str((tmp_path / "web" / "app.TS").resolve()),
            language="TypeScript",
            content="export const answer = 42;\n",
            line_count=1,
        ),
    ]


@pytest.mark.parametrize(
    "directory",
    [
        ".git",
        "node_modules",
        ".venv",
        "dist",
        "build",
        "target",
        "__pycache__",
        "coverage",
    ],
)
def test_scan_ignores_generated_directories(
    tmp_path: Path, directory: str
) -> None:
    ignored_path = tmp_path / directory
    ignored_path.mkdir()
    (ignored_path / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    (tmp_path / "included.py").write_text("included = True\n", encoding="utf-8")

    files = FileScanner().scan(tmp_path)

    assert [file.relative_path for file in files] == ["included.py"]


def test_scan_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02")
    (tmp_path / "invalid.py").write_bytes(b"\xff\xfe")

    assert FileScanner().scan(tmp_path) == []


def test_scan_skips_files_larger_than_one_megabyte(tmp_path: Path) -> None:
    (tmp_path / "large.py").write_bytes(b"x" * (FileScanner.MAX_FILE_SIZE + 1))
    (tmp_path / "allowed.py").write_text("x = 1\n", encoding="utf-8")

    files = FileScanner().scan(tmp_path)

    assert [file.relative_path for file in files] == ["allowed.py"]


def test_scan_rejects_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        FileScanner().scan(file_path)
