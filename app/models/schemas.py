from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class RepoIndexRequest(BaseModel):
    repo_url: HttpUrl

    @field_validator("repo_url")
    @classmethod
    def validate_github_url(cls, value: HttpUrl) -> HttpUrl:
        if value.host not in {"github.com", "www.github.com"}:
            raise ValueError("repo_url must point to github.com")
        return value


class RepoIndexResponse(BaseModel):
    repo_id: str
    files_scanned: int
    chunks_created: int
    indexing_time_seconds: float


class RepositoryMetadata(BaseModel):
    repo_id: str
    repo_url: str
    local_path: str
    indexed_at: datetime
    file_count: int
    chunk_count: int


class QueryRequest(BaseModel):
    repo_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)

    @field_validator("repo_id", "question")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class SourceCitation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class QueryHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repo_id: str | None
    question: str
    answer: str | None
    duration_ms: float | None
    created_at: datetime
