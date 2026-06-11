from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    absolute_path: str
    language: str
    content: str
    line_count: int


class FileScanner:
    MAX_FILE_SIZE = 1024 * 1024
    BINARY_SAMPLE_SIZE = 8192

    IGNORED_DIRECTORIES = frozenset(
        {
            ".git",
            "node_modules",
            ".venv",
            "dist",
            "build",
            "target",
            "__pycache__",
            "coverage",
        }
    )

    LANGUAGE_BY_EXTENSION = {
        ".bash": "Shell",
        ".c": "C",
        ".cc": "C++",
        ".clj": "Clojure",
        ".cljs": "Clojure",
        ".cpp": "C++",
        ".cs": "C#",
        ".css": "CSS",
        ".cxx": "C++",
        ".dart": "Dart",
        ".erl": "Erlang",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".fish": "Shell",
        ".fs": "F#",
        ".fsx": "F#",
        ".go": "Go",
        ".groovy": "Groovy",
        ".h": "C",
        ".hh": "C++",
        ".hpp": "C++",
        ".hrl": "Erlang",
        ".html": "HTML",
        ".htm": "HTML",
        ".java": "Java",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".kt": "Kotlin",
        ".kts": "Kotlin",
        ".less": "Less",
        ".lua": "Lua",
        ".m": "Objective-C",
        ".mm": "Objective-C++",
        ".php": "PHP",
        ".pl": "Perl",
        ".pm": "Perl",
        ".py": "Python",
        ".r": "R",
        ".rb": "Ruby",
        ".rs": "Rust",
        ".sass": "Sass",
        ".scala": "Scala",
        ".scss": "SCSS",
        ".sh": "Shell",
        ".sql": "SQL",
        ".svelte": "Svelte",
        ".swift": "Swift",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".vue": "Vue",
        ".zsh": "Shell",
    }

    def __init__(
        self,
        ignored_directories: set[str] | frozenset[str] | None = None,
        max_file_size: int = MAX_FILE_SIZE,
    ) -> None:
        if max_file_size < 0:
            raise ValueError("max_file_size must be non-negative")

        self.ignored_directories = frozenset(
            ignored_directories
            if ignored_directories is not None
            else self.IGNORED_DIRECTORIES
        )
        self.max_file_size = max_file_size

    def scan(self, repository_path: str | Path) -> list[SourceFile]:
        root = Path(repository_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Repository path is not a directory: {root}")

        source_files: list[SourceFile] = []

        for current_root, directories, filenames in os.walk(root):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in self.ignored_directories
            )

            current_path = Path(current_root)
            for filename in sorted(filenames):
                path = current_path / filename
                language = self.detect_language(path)
                if language is None:
                    continue

                source_file = self._read_source_file(root, path, language)
                if source_file is not None:
                    source_files.append(source_file)

        return source_files

    def detect_language(self, path: str | Path) -> str | None:
        return self.LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower())

    def _read_source_file(
        self, root: Path, path: Path, language: str
    ) -> SourceFile | None:
        try:
            if not path.is_file() or path.stat().st_size > self.max_file_size:
                return None

            with path.open("rb") as file_handle:
                sample = file_handle.read(self.BINARY_SAMPLE_SIZE)
                if b"\x00" in sample:
                    return None

                remaining = file_handle.read()
                content = (sample + remaining).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        return SourceFile(
            relative_path=path.relative_to(root).as_posix(),
            absolute_path=str(path.resolve()),
            language=language,
            content=content,
            line_count=len(content.splitlines()),
        )
