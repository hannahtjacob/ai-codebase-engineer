from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import networkx as nx

from app.core.ast_parser import ParsedPythonFile, PythonASTParser
from app.core.file_scanner import SourceFile


@dataclass(frozen=True)
class GraphData:
    nodes: list[dict[str, object]]
    edges: list[dict[str, str]]


class GraphBuilder:
    def __init__(self, parser: PythonASTParser | None = None) -> None:
        self.parser = parser or PythonASTParser()

    def build(
        self,
        repo_id: str,
        files: Iterable[SourceFile],
    ) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        repository_id = f"repository:{repo_id}"
        graph.add_node(
            repository_id,
            type="Repository",
            name=repo_id,
            file_path=None,
            line=None,
        )

        parsed_files: list[ParsedPythonFile] = []
        for source_file in files:
            if source_file.language.lower() != "python":
                continue
            try:
                parsed = self.parser.parse_file(source_file)
            except SyntaxError:
                continue
            parsed_files.append(parsed)
            self._add_parsed_file(graph, repository_id, parsed)

        self._add_import_edges(graph, parsed_files)
        self._add_call_edges(graph, parsed_files)
        return graph

    def build_from_path(
        self,
        repo_id: str,
        repository_path: str | Path,
    ) -> nx.MultiDiGraph:
        from app.core.file_scanner import FileScanner

        return self.build(repo_id, FileScanner().scan(repository_path))

    @staticmethod
    def to_data(graph: nx.MultiDiGraph) -> GraphData:
        nodes = [
            {
                "id": node_id,
                "type": attributes["type"],
                "name": attributes["name"],
                "file_path": attributes.get("file_path"),
                "line": attributes.get("line"),
            }
            for node_id, attributes in sorted(graph.nodes(data=True))
        ]
        edges = [
            {
                "source": source,
                "target": target,
                "type": attributes["type"],
            }
            for source, target, _, attributes in sorted(
                graph.edges(keys=True, data=True),
                key=lambda edge: (
                    edge[0],
                    edge[1],
                    edge[3]["type"],
                    edge[2],
                ),
            )
        ]
        return GraphData(nodes=nodes, edges=edges)

    def build_data(
        self,
        repo_id: str,
        files: Iterable[SourceFile],
    ) -> GraphData:
        return self.to_data(self.build(repo_id, files))

    @staticmethod
    def _add_parsed_file(
        graph: nx.MultiDiGraph,
        repository_id: str,
        parsed: ParsedPythonFile,
    ) -> None:
        for node in parsed.nodes:
            graph.add_node(
                node.node_id,
                type=node.node_type,
                name=node.name,
                file_path=node.file_path,
                line=node.line,
            )
        graph.add_edge(
            repository_id,
            PythonASTParser.file_node_id(parsed.file_path),
            type="CONTAINS",
        )
        for edge in parsed.edges:
            graph.add_edge(edge.source, edge.target, type=edge.edge_type)

    def _add_import_edges(
        self,
        graph: nx.MultiDiGraph,
        parsed_files: list[ParsedPythonFile],
    ) -> None:
        known_files = {parsed.file_path for parsed in parsed_files}
        for parsed in parsed_files:
            source_id = self.parser.file_node_id(parsed.file_path)
            for import_ref in parsed.imports:
                target_path = self._resolve_import(
                    parsed.file_path,
                    import_ref.module,
                    import_ref.imported_name,
                    import_ref.level,
                    known_files,
                )
                target_id = (
                    self.parser.file_node_id(target_path)
                    if target_path is not None
                    else import_ref.node_id
                )
                graph.add_edge(source_id, target_id, type="IMPORTS")

    @staticmethod
    def _resolve_import(
        source_path: str,
        module: str,
        imported_name: str | None,
        level: int,
        known_files: set[str],
    ) -> str | None:
        module_parts = [part for part in module.split(".") if part]
        if level:
            source_parts = list(PurePosixPath(source_path).parent.parts)
            keep = max(0, len(source_parts) - (level - 1))
            module_parts = source_parts[:keep] + module_parts

        candidates: list[str] = []
        if module_parts:
            base = "/".join(module_parts)
            candidates.extend((f"{base}.py", f"{base}/__init__.py"))
            if imported_name:
                candidates.insert(0, f"{base}/{imported_name}.py")
        elif imported_name:
            candidates.append(f"{imported_name}.py")

        return next(
            (candidate for candidate in candidates if candidate in known_files),
            None,
        )

    @staticmethod
    def _add_call_edges(
        graph: nx.MultiDiGraph,
        parsed_files: list[ParsedPythonFile],
    ) -> None:
        symbols_by_full_name: dict[str, list[str]] = {}
        symbols_by_short_name: dict[str, list[str]] = {}
        for parsed in parsed_files:
            for node in parsed.nodes:
                if node.node_type != "Function":
                    continue
                symbols_by_full_name.setdefault(node.name, []).append(node.node_id)
                symbols_by_short_name.setdefault(
                    node.name.rsplit(".", 1)[-1], []
                ).append(node.node_id)

        for parsed in parsed_files:
            aliases = GraphBuilder._import_aliases(parsed)
            for call in parsed.calls:
                call_name = GraphBuilder._expand_call_alias(call.name, aliases)
                candidates = symbols_by_full_name.get(call_name, [])
                if not candidates:
                    candidates = symbols_by_short_name.get(
                        call_name.rsplit(".", 1)[-1], []
                    )
                target = GraphBuilder._choose_call_target(
                    call.caller_id, candidates
                )
                if target is not None:
                    graph.add_edge(call.caller_id, target, type="CALLS")

    @staticmethod
    def _import_aliases(parsed: ParsedPythonFile) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for import_ref in parsed.imports:
            if import_ref.imported_name:
                local_name = import_ref.alias or import_ref.imported_name
                aliases[local_name] = import_ref.imported_name
            elif import_ref.module:
                local_name = import_ref.alias or import_ref.module.split(".", 1)[0]
                aliases[local_name] = import_ref.module
        return aliases

    @staticmethod
    def _expand_call_alias(name: str, aliases: dict[str, str]) -> str:
        first, separator, remainder = name.partition(".")
        replacement = aliases.get(first)
        if replacement is None:
            return name
        return replacement + (f".{remainder}" if separator else "")

    @staticmethod
    def _choose_call_target(
        caller_id: str,
        candidates: list[str],
    ) -> str | None:
        if not candidates:
            return None
        caller_file = caller_id.split(":", 2)[1]
        same_file = [
            candidate
            for candidate in candidates
            if candidate.split(":", 2)[1] == caller_file
        ]
        if len(same_file) == 1:
            return same_file[0]
        if len(candidates) == 1:
            return candidates[0]
        return None
