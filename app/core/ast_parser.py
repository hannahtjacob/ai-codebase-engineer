from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from app.core.file_scanner import SourceFile


@dataclass(frozen=True)
class ASTNode:
    node_id: str
    node_type: str
    name: str
    file_path: str
    line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class ASTEdge:
    source: str
    target: str
    edge_type: str


@dataclass(frozen=True)
class ImportReference:
    node_id: str
    module: str
    imported_name: str | None
    alias: str | None
    level: int
    line: int


@dataclass(frozen=True)
class CallReference:
    caller_id: str
    name: str
    line: int


@dataclass(frozen=True)
class ParsedPythonFile:
    file_path: str
    nodes: tuple[ASTNode, ...]
    edges: tuple[ASTEdge, ...]
    imports: tuple[ImportReference, ...]
    calls: tuple[CallReference, ...]


class PythonASTParser:
    def parse_file(self, source_file: SourceFile) -> ParsedPythonFile:
        return self.parse(source_file.content, source_file.relative_path)

    def parse(self, content: str, file_path: str | Path) -> ParsedPythonFile:
        normalized_path = Path(file_path).as_posix()
        tree = ast.parse(content, filename=normalized_path)
        file_id = self.file_node_id(normalized_path)

        nodes: list[ASTNode] = [
            ASTNode(
                node_id=file_id,
                node_type="File",
                name=normalized_path,
                file_path=normalized_path,
            )
        ]
        edges: list[ASTEdge] = []
        imports: list[ImportReference] = []
        calls: list[CallReference] = []

        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._extract_imports(
                    statement,
                    normalized_path,
                    file_id,
                    nodes,
                    edges,
                    imports,
                )
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_id = self.symbol_node_id(
                    normalized_path, "Function", statement.name
                )
                nodes.append(
                    self._symbol_node(
                        function_id,
                        "Function",
                        statement.name,
                        normalized_path,
                        statement,
                    )
                )
                edges.append(ASTEdge(file_id, function_id, "DEFINES"))
                calls.extend(self._extract_calls(statement, function_id))
            elif isinstance(statement, ast.ClassDef):
                class_id = self.symbol_node_id(
                    normalized_path, "Class", statement.name
                )
                nodes.append(
                    self._symbol_node(
                        class_id,
                        "Class",
                        statement.name,
                        normalized_path,
                        statement,
                    )
                )
                edges.append(ASTEdge(file_id, class_id, "DEFINES"))
                self._extract_methods(
                    statement,
                    normalized_path,
                    class_id,
                    nodes,
                    edges,
                    calls,
                )

        return ParsedPythonFile(
            file_path=normalized_path,
            nodes=tuple(nodes),
            edges=tuple(edges),
            imports=tuple(imports),
            calls=tuple(calls),
        )

    def _extract_imports(
        self,
        statement: ast.Import | ast.ImportFrom,
        file_path: str,
        file_id: str,
        nodes: list[ASTNode],
        edges: list[ASTEdge],
        imports: list[ImportReference],
    ) -> None:
        module = statement.module or "" if isinstance(statement, ast.ImportFrom) else ""
        level = statement.level if isinstance(statement, ast.ImportFrom) else 0

        for index, alias in enumerate(statement.names):
            imported_module = alias.name if isinstance(statement, ast.Import) else module
            imported_name = alias.name if isinstance(statement, ast.ImportFrom) else None
            display_name = (
                f"{'.' * level}{module}.{alias.name}".strip(".")
                if isinstance(statement, ast.ImportFrom)
                else alias.name
            )
            import_id = (
                f"import:{file_path}:{statement.lineno}:{index}:{display_name}"
            )
            nodes.append(
                ASTNode(
                    node_id=import_id,
                    node_type="Import",
                    name=display_name,
                    file_path=file_path,
                    line=statement.lineno,
                    end_line=getattr(statement, "end_lineno", statement.lineno),
                )
            )
            edges.append(ASTEdge(file_id, import_id, "DEFINES"))
            imports.append(
                ImportReference(
                    node_id=import_id,
                    module=imported_module,
                    imported_name=imported_name,
                    alias=alias.asname,
                    level=level,
                    line=statement.lineno,
                )
            )

    def _extract_methods(
        self,
        class_node: ast.ClassDef,
        file_path: str,
        class_id: str,
        nodes: list[ASTNode],
        edges: list[ASTEdge],
        calls: list[CallReference],
    ) -> None:
        for statement in class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualified_name = f"{class_node.name}.{statement.name}"
            method_id = self.symbol_node_id(
                file_path, "Function", qualified_name
            )
            nodes.append(
                self._symbol_node(
                    method_id,
                    "Function",
                    qualified_name,
                    file_path,
                    statement,
                )
            )
            edges.append(ASTEdge(class_id, method_id, "CONTAINS"))
            calls.extend(self._extract_calls(statement, method_id))

    @staticmethod
    def _extract_calls(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        caller_id: str,
    ) -> list[CallReference]:
        visitor = _CallVisitor(caller_id)
        for statement in node.body:
            visitor.visit(statement)
        return visitor.calls

    @staticmethod
    def _call_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = PythonASTParser._call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return None

    @staticmethod
    def _symbol_node(
        node_id: str,
        node_type: str,
        name: str,
        file_path: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> ASTNode:
        return ASTNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            file_path=file_path,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )

    @staticmethod
    def file_node_id(file_path: str) -> str:
        return f"file:{file_path}"

    @staticmethod
    def symbol_node_id(file_path: str, node_type: str, name: str) -> str:
        return f"{node_type.lower()}:{file_path}:{name}"


ASTParser = PythonASTParser


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, caller_id: str) -> None:
        self.caller_id = caller_id
        self.calls: list[CallReference] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = PythonASTParser._call_name(node.func)
        if name:
            self.calls.append(
                CallReference(
                    caller_id=self.caller_id,
                    name=name,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None
