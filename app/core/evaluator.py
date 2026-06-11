from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from app.core.cache import SQLiteCache
from app.core.rag_engine import RagEngine
from app.core.retriever import RetrievedChunk, Retriever


CITATION_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+):\d+-\d+"
)


@dataclass(frozen=True)
class EvaluationCase:
    repo_url: str
    question: str
    expected_files: tuple[str, ...]
    repo_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationCase:
        repo_url = value.get("repo_url")
        question = value.get("question")
        expected_files = value.get("expected_files")
        if not isinstance(repo_url, str) or not repo_url.strip():
            raise ValueError("repo_url must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(expected_files, list) or not all(
            isinstance(path, str) and path.strip() for path in expected_files
        ):
            raise ValueError("expected_files must be a list of non-empty strings")
        repo_id = value.get("repo_id")
        if repo_id is not None and not isinstance(repo_id, str):
            raise ValueError("repo_id must be a string when provided")
        return cls(
            repo_url=repo_url.strip(),
            question=question.strip(),
            expected_files=tuple(
                normalize_file_path(path) for path in expected_files
            ),
            repo_id=repo_id,
        )


@dataclass(frozen=True)
class QuestionEvaluation:
    repo_url: str
    repo_id: str
    question: str
    expected_files: list[str]
    retrieved_files_at_5: list[str]
    retrieved_files_at_10: list[str]
    cited_files: list[str]
    recall_at_5: float
    recall_at_10: float
    citation_file_accuracy: float
    latency_ms: float
    chunks_retrieved: int
    answer: str


@dataclass(frozen=True)
class EvaluationSummary:
    questions: int
    recall_at_5: float
    recall_at_10: float
    citation_file_accuracy: float
    average_latency_ms: float
    average_chunks_retrieved: float
    cache_hit_rate: float | None


@dataclass(frozen=True)
class EvaluationReport:
    summary: EvaluationSummary
    results: list[QuestionEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": asdict(self.summary),
            "results": [asdict(result) for result in self.results],
        }


RepoIdResolver = Callable[[EvaluationCase], str]


class Evaluator:
    def __init__(
        self,
        retriever: Retriever,
        rag_engine: RagEngine,
        *,
        cache: SQLiteCache | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.retriever = retriever
        self.rag_engine = rag_engine
        self.cache = cache
        self._clock = clock

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        repo_id_resolver: RepoIdResolver,
    ) -> EvaluationReport:
        if not cases:
            raise ValueError("at least one evaluation case is required")
        if self.cache is not None:
            self.cache.reset_stats()

        results = [
            self.evaluate_case(case, repo_id_resolver(case))
            for case in cases
        ]
        cache_hit_rate = (
            self.cache.stats().hit_rate if self.cache is not None else None
        )
        summary = EvaluationSummary(
            questions=len(results),
            recall_at_5=mean(result.recall_at_5 for result in results),
            recall_at_10=mean(result.recall_at_10 for result in results),
            citation_file_accuracy=mean(
                result.citation_file_accuracy for result in results
            ),
            average_latency_ms=mean(result.latency_ms for result in results),
            average_chunks_retrieved=mean(
                result.chunks_retrieved for result in results
            ),
            cache_hit_rate=cache_hit_rate,
        )
        return EvaluationReport(summary=summary, results=results)

    def evaluate_case(
        self,
        case: EvaluationCase,
        repo_id: str,
    ) -> QuestionEvaluation:
        started_at = self._clock()
        chunks = self._retrieve(repo_id, case.question, top_k=10)
        rag_result = self.rag_engine.answer_with_sources(
            repo_id,
            case.question,
            k=10,
        )
        latency_ms = (self._clock() - started_at) * 1000

        expected = set(case.expected_files)
        files_at_5 = unique_files(chunks[:5])
        files_at_10 = unique_files(chunks[:10])
        cited_files = extract_cited_files(rag_result.answer)

        return QuestionEvaluation(
            repo_url=case.repo_url,
            repo_id=repo_id,
            question=case.question,
            expected_files=list(case.expected_files),
            retrieved_files_at_5=files_at_5,
            retrieved_files_at_10=files_at_10,
            cited_files=cited_files,
            recall_at_5=recall(expected, set(files_at_5)),
            recall_at_10=recall(expected, set(files_at_10)),
            citation_file_accuracy=citation_accuracy(
                expected,
                set(cited_files),
            ),
            latency_ms=latency_ms,
            chunks_retrieved=len(chunks),
            answer=rag_result.answer,
        )

    def _retrieve(
        self,
        repo_id: str,
        question: str,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        graph_retrieve = getattr(
            self.retriever,
            "retrieve_with_graph_expansion",
            None,
        )
        if callable(graph_retrieve):
            return graph_retrieve(repo_id, question, top_k=top_k)
        return self.retriever.retrieve(repo_id, question, k=top_k)


def normalize_file_path(path: str) -> str:
    return path.strip().replace("\\", "/").removeprefix("./")


def unique_files(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return list(
        dict.fromkeys(normalize_file_path(chunk.file_path) for chunk in chunks)
    )


def extract_cited_files(answer: str) -> list[str]:
    return list(
        dict.fromkeys(
            normalize_file_path(match.group("path"))
            for match in CITATION_PATTERN.finditer(answer)
        )
    )


def recall(expected: set[str], retrieved: set[str]) -> float:
    return len(expected & retrieved) / len(expected) if expected else 1.0


def citation_accuracy(expected: set[str], cited: set[str]) -> float:
    if not cited:
        return 0.0
    return len(expected & cited) / len(cited)


def markdown_table(report: EvaluationReport) -> str:
    summary = report.summary
    cache_rate = (
        f"{summary.cache_hit_rate:.2%}"
        if summary.cache_hit_rate is not None
        else "N/A"
    )
    rows = [
        ("Questions", str(summary.questions)),
        ("Recall@5", f"{summary.recall_at_5:.2%}"),
        ("Recall@10", f"{summary.recall_at_10:.2%}"),
        (
            "Citation file accuracy",
            f"{summary.citation_file_accuracy:.2%}",
        ),
        ("Average latency", f"{summary.average_latency_ms:.2f} ms"),
        (
            "Average chunks retrieved",
            f"{summary.average_chunks_retrieved:.2f}",
        ),
        ("Cache hit rate", cache_rate),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {metric} | {value} |" for metric, value in rows)
    return "\n".join(lines)
