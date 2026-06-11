from app.core.evaluator import (
    EvaluationCase,
    Evaluator,
    extract_cited_files,
    markdown_table,
)
from app.core.rag_engine import RagResult
from app.core.retriever import RetrievedChunk


def make_chunk(chunk_id: str, file_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        repo_id="repo-123",
        file_path=file_path,
        language="Python",
        start_line=1,
        end_line=10,
        symbol_name=None,
        symbol_type=None,
        content="pass\n",
        distance=0.1,
    )


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve_with_graph_expansion(
        self,
        repo_id: str,
        question: str,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        assert repo_id == "repo-123"
        assert question
        return self.chunks[:top_k]


class StubRagEngine:
    def __init__(self, answer: str, sources: list[RetrievedChunk]) -> None:
        self.result = RagResult(answer=answer, sources=tuple(sources))

    def answer_with_sources(
        self,
        repo_id: str,
        question: str,
        k: int = 8,
    ) -> RagResult:
        return self.result


def test_evaluator_computes_retrieval_citation_and_latency_metrics() -> None:
    chunks = [
        make_chunk("1", "app/auth/routes.py"),
        make_chunk("2", "app/other.py"),
        make_chunk("3", "app/auth/utils.py"),
    ]
    ticks = iter([10.0, 10.125])
    evaluator = Evaluator(
        StubRetriever(chunks),  # type: ignore[arg-type]
        StubRagEngine(  # type: ignore[arg-type]
            "See app/auth/routes.py:1-10 and app/wrong.py:1-10.",
            chunks,
        ),
        clock=lambda: next(ticks),
    )
    case = EvaluationCase(
        repo_url="https://github.com/example/repo",
        repo_id="repo-123",
        question="Where is authentication?",
        expected_files=("app/auth/routes.py", "app/auth/utils.py"),
    )

    report = evaluator.evaluate([case], lambda item: item.repo_id or "")

    result = report.results[0]
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.citation_file_accuracy == 0.5
    assert result.latency_ms == 125.0
    assert result.chunks_retrieved == 3
    assert report.summary.cache_hit_rate is None
    assert "| Recall@5 | 100.00% |" in markdown_table(report)


def test_extract_cited_files_deduplicates_paths() -> None:
    answer = (
        "See `app/auth/routes.py:10-45` and app/auth/routes.py:50-60, "
        "then app/models/user.py:1-40."
    )

    assert extract_cited_files(answer) == [
        "app/auth/routes.py",
        "app/models/user.py",
    ]
