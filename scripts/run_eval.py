from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.cache import SQLiteCache
from app.config import load_environment
from app.core.embedding_service import EmbeddingService
from app.core.evaluator import EvaluationCase, Evaluator, markdown_table
from app.core.graph_builder import GraphBuilder
from app.core.indexer import RepositoryIndexer
from app.core.rag_engine import RagEngine
from app.core.repo_loader import RepoLoader
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.models.db import get_engine, get_session_factory, init_db


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row must be a JSON object")
                cases.append(EvaluationCase.from_dict(value))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid evaluation row at {input_path}:{line_number}: {error}"
                ) from error
    if not cases:
        raise ValueError(f"No evaluation cases found in {input_path}")
    return cases


def save_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(
        description="Evaluate repository retrieval and codebase QA quality."
    )
    parser.add_argument(
        "--questions",
        default="data/eval/questions.jsonl",
        help="Path to evaluation questions in JSONL format",
    )
    parser.add_argument(
        "--output",
        default="data/eval/results.json",
        help="Path for the JSON evaluation report",
    )
    args = parser.parse_args()

    cases = load_cases(args.questions)
    engine = init_db(get_engine())
    session_factory = get_session_factory(engine)
    cache = SQLiteCache(engine)
    embedding_service = EmbeddingService(cache=cache)
    vector_store = VectorStore()
    graph_builder = GraphBuilder()
    repo_loader = RepoLoader(os.getenv("REPO_STORAGE_PATH", "data/repos"))
    indexer = RepositoryIndexer(
        session_factory,
        repo_loader=repo_loader,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        graph_builder=graph_builder,
        session_factory=session_factory,
    )
    rag_engine = RagEngine(retriever=retriever, cache=cache)
    evaluator = Evaluator(retriever, rag_engine, cache=cache)

    repo_ids: dict[str, str] = {}

    def resolve_repo_id(case: EvaluationCase) -> str:
        if case.repo_id:
            return case.repo_id
        if case.repo_url not in repo_ids:
            result = indexer.index_url(case.repo_url)
            repo_ids[case.repo_url] = result.repo_id
        return repo_ids[case.repo_url]

    report = evaluator.evaluate(cases, resolve_repo_id)
    save_report(args.output, report.to_dict())
    print(markdown_table(report))
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
