from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Iterable

from app.core.file_scanner import SourceFile


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    repo_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    symbol_type: str | None
    content: str


class CodeChunker:
    WINDOW_SIZE = 80
    WINDOW_OVERLAP = 20

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        window_overlap: int = WINDOW_OVERLAP,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if window_overlap < 0 or window_overlap >= window_size:
            raise ValueError(
                "window_overlap must be non-negative and smaller than window_size"
            )

        self.window_size = window_size
        self.window_overlap = window_overlap

    def chunk_files(
        self, files: Iterable[SourceFile], repo_id: str
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for source_file in files:
            chunks.extend(self.chunk_file(source_file, repo_id))
        return chunks

    def chunk_file(self, source_file: SourceFile, repo_id: str) -> list[CodeChunk]:
        if source_file.language.lower() == "python":
            try:
                return self._chunk_python(source_file, repo_id)
            except SyntaxError:
                pass

        return self._chunk_with_sliding_window(source_file, repo_id)

    def _chunk_python(
        self, source_file: SourceFile, repo_id: str
    ) -> list[CodeChunk]:
        tree = ast.parse(source_file.content)
        lines = source_file.content.splitlines(keepends=True)
        chunks: list[CodeChunk] = []

        for node in tree.body:
            symbol_type = self._python_symbol_type(node)
            if symbol_type is None:
                continue

            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                continue

            start_line = self._python_start_line(node)
            content = "".join(lines[start_line - 1 : end_line])
            chunks.append(
                self._make_chunk(
                    repo_id=repo_id,
                    source_file=source_file,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=node.name,
                    symbol_type=symbol_type,
                    content=content,
                )
            )

        return chunks

    def _chunk_with_sliding_window(
        self, source_file: SourceFile, repo_id: str
    ) -> list[CodeChunk]:
        lines = source_file.content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[CodeChunk] = []
        step = self.window_size - self.window_overlap

        for start_index in range(0, len(lines), step):
            end_index = min(start_index + self.window_size, len(lines))
            chunks.append(
                self._make_chunk(
                    repo_id=repo_id,
                    source_file=source_file,
                    start_line=start_index + 1,
                    end_line=end_index,
                    symbol_name=None,
                    symbol_type=None,
                    content="".join(lines[start_index:end_index]),
                )
            )
            if end_index == len(lines):
                break

        return chunks

    @staticmethod
    def _python_symbol_type(node: ast.AST) -> str | None:
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function"
        if isinstance(node, ast.FunctionDef):
            return "function"
        if isinstance(node, ast.ClassDef):
            return "class"
        return None

    @staticmethod
    def _python_start_line(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> int:
        decorator_lines = [
            decorator.lineno
            for decorator in node.decorator_list
            if hasattr(decorator, "lineno")
        ]
        return min([node.lineno, *decorator_lines])

    def _make_chunk(
        self,
        *,
        repo_id: str,
        source_file: SourceFile,
        start_line: int,
        end_line: int,
        symbol_name: str | None,
        symbol_type: str | None,
        content: str,
    ) -> CodeChunk:
        chunk_id = self._generate_chunk_id(
            repo_id=repo_id,
            file_path=source_file.relative_path,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
        )
        return CodeChunk(
            chunk_id=chunk_id,
            repo_id=repo_id,
            file_path=source_file.relative_path,
            language=source_file.language,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            content=content,
        )

    @staticmethod
    def _generate_chunk_id(
        *,
        repo_id: str,
        file_path: str,
        start_line: int,
        end_line: int,
        symbol_name: str | None,
        symbol_type: str | None,
    ) -> str:
        identity = "\0".join(
            (
                repo_id,
                file_path,
                str(start_line),
                str(end_line),
                symbol_name or "",
                symbol_type or "",
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
