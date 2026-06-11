from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.core.embedding_service import EmbeddingService
from app.core.graph_builder import GraphBuilder
from app.models.db import CodeChunk as CodeChunkRecord
from app.models.db import Repository
from app.core.vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    repo_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    symbol_type: str | None
    content: str
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


GraphProvider = Callable[[str], nx.MultiDiGraph]
ChunkLoader = Callable[
    [str, Sequence[str], nx.MultiDiGraph],
    list[RetrievedChunk],
]


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        *,
        graph_builder: GraphBuilder | None = None,
        session_factory: sessionmaker | None = None,
        graph_provider: GraphProvider | None = None,
        chunk_loader: ChunkLoader | None = None,
    ) -> None:
        if (graph_provider is None) != (chunk_loader is None):
            raise ValueError(
                "graph_provider and chunk_loader must be provided together"
            )

        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.graph_builder = graph_builder or GraphBuilder()
        self.session_factory = session_factory
        self._graph_provider = graph_provider
        self._chunk_loader = chunk_loader

        if graph_provider is None and session_factory is not None:
            self._graph_provider = self._build_repository_graph
            self._chunk_loader = self._load_chunks_for_nodes

    def retrieve(
        self,
        repo_id: str,
        query: str,
        k: int = 8,
    ) -> list[RetrievedChunk]:
        if not repo_id.strip():
            raise ValueError("repo_id must not be empty")
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        [query_embedding] = self.embedding_service.embed_texts([query])
        results = self.vector_store.search(repo_id, query_embedding, k=k)
        return [self._to_retrieved_chunk(result) for result in results]

    def retrieve_with_graph_expansion(
        self,
        repo_id: str,
        question: str,
        top_k: int = 8,
        expansion_depth: int = 1,
    ) -> list[RetrievedChunk]:
        if expansion_depth < 0:
            raise ValueError("expansion_depth must be non-negative")

        seeds = self.retrieve(repo_id, question, k=top_k)
        if (
            not seeds
            or expansion_depth == 0
            or self._graph_provider is None
            or self._chunk_loader is None
        ):
            return seeds

        graph = self._graph_provider(repo_id)
        seed_node_ids = self._seed_graph_nodes(graph, seeds)
        expanded_node_ids = self._expand_graph_nodes(
            graph,
            seed_node_ids,
            expansion_depth,
        )
        expanded_chunks = self._chunk_loader(
            repo_id,
            expanded_node_ids,
            graph,
        )
        return self._merge_ranked(seeds, expanded_chunks)

    @staticmethod
    def _seed_graph_nodes(
        graph: nx.MultiDiGraph,
        seeds: Sequence[RetrievedChunk],
    ) -> list[str]:
        node_ids: list[str] = []
        for seed in seeds:
            file_id = f"file:{seed.file_path}"
            if file_id in graph:
                node_ids.append(file_id)

            if seed.symbol_name is None:
                continue
            for node_id, attributes in graph.nodes(data=True):
                if attributes.get("file_path") != seed.file_path:
                    continue
                if attributes.get("type") not in {"Function", "Class"}:
                    continue
                name = str(attributes.get("name", ""))
                if name == seed.symbol_name or name.rsplit(".", 1)[-1] == seed.symbol_name:
                    node_ids.append(node_id)
        return list(dict.fromkeys(node_ids))

    @staticmethod
    def _expand_graph_nodes(
        graph: nx.MultiDiGraph,
        seed_node_ids: Sequence[str],
        expansion_depth: int,
    ) -> list[str]:
        visited = set(seed_node_ids)
        frontier = list(seed_node_ids)
        expanded: list[str] = []

        for _ in range(expansion_depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                neighbors: set[str] = set()
                for _, target, attributes in graph.out_edges(node_id, data=True):
                    if attributes.get("type") in {"IMPORTS", "CALLS"}:
                        neighbors.add(target)
                for source, _, attributes in graph.in_edges(node_id, data=True):
                    if attributes.get("type") == "CONTAINS":
                        neighbors.add(source)

                for neighbor in sorted(neighbors):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    expanded.append(neighbor)
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return expanded

    @staticmethod
    def _merge_ranked(
        seeds: Sequence[RetrievedChunk],
        expanded: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        merged = list(seeds)
        seen = {chunk.chunk_id for chunk in seeds}
        base_distance = max((chunk.distance for chunk in seeds), default=0.0) + 1.0

        for rank, chunk in enumerate(expanded):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            merged.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    repo_id=chunk.repo_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    content=chunk.content,
                    distance=base_distance + rank,
                )
            )
        return merged

    def _build_repository_graph(self, repo_id: str) -> nx.MultiDiGraph:
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for graph retrieval")
        with self.session_factory() as session:
            repository = session.get(Repository, repo_id)
            if repository is None:
                raise ValueError(f"Repository not found: {repo_id}")
            repository_path = Path(repository.local_path)
        if not repository_path.is_dir():
            raise ValueError(
                f"Repository checkout is unavailable: {repository_path}"
            )
        return self.graph_builder.build_from_path(repo_id, repository_path)

    def _load_chunks_for_nodes(
        self,
        repo_id: str,
        node_ids: Sequence[str],
        graph: nx.MultiDiGraph,
    ) -> list[RetrievedChunk]:
        if self.session_factory is None or not node_ids:
            return []

        with self.session_factory() as session:
            records = session.scalars(
                select(CodeChunkRecord)
                .where(CodeChunkRecord.repository_id == repo_id)
                .order_by(
                    CodeChunkRecord.file_path,
                    CodeChunkRecord.start_line,
                    CodeChunkRecord.chunk_id,
                )
            ).all()

        records_by_file: dict[str, list[CodeChunkRecord]] = {}
        for record in records:
            records_by_file.setdefault(record.file_path, []).append(record)

        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for node_id in node_ids:
            attributes = graph.nodes[node_id]
            file_path = attributes.get("file_path")
            if not isinstance(file_path, str):
                continue
            node_type = attributes.get("type")
            node_name = str(attributes.get("name", ""))

            for record in records_by_file.get(file_path, []):
                if node_type in {"Function", "Class"}:
                    short_name = node_name.rsplit(".", 1)[-1]
                    parent_class = (
                        node_name.rsplit(".", 1)[0]
                        if node_type == "Function" and "." in node_name
                        else None
                    )
                    if record.symbol_name not in {
                        node_name,
                        short_name,
                        parent_class,
                    }:
                        continue
                if record.chunk_id in seen:
                    continue
                seen.add(record.chunk_id)
                chunks.append(self._from_record(record))
        return chunks

    @staticmethod
    def _to_retrieved_chunk(result: SearchResult) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=result.chunk_id,
            repo_id=result.repo_id,
            file_path=result.file_path,
            language=result.language,
            start_line=result.start_line,
            end_line=result.end_line,
            symbol_name=result.symbol_name,
            symbol_type=result.symbol_type,
            content=result.content,
            distance=result.distance,
        )

    @staticmethod
    def _from_record(record: CodeChunkRecord) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=record.chunk_id,
            repo_id=record.repository_id,
            file_path=record.file_path,
            language=record.language,
            start_line=record.start_line,
            end_line=record.end_line,
            symbol_name=record.symbol_name,
            symbol_type=record.symbol_type,
            content=record.content,
            distance=0.0,
        )
