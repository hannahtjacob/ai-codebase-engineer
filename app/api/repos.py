from __future__ import annotations

import logging
import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from git.exc import GitError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_indexer, get_session
from app.config import is_development
from app.core.embedding_service import EmbeddingServiceError
from app.core.indexer import RepositoryIndexer
from app.models.db import CodeChunk, Repository, SourceFile
from app.models.schemas import (
    RepoIndexRequest,
    RepoIndexResponse,
    RepositoryMetadata,
)


router = APIRouter(prefix="/repos", tags=["repositories"])
logger = logging.getLogger(__name__)


def _error_detail(prefix: str, error: Exception) -> str:
    if not is_development():
        return prefix

    message = str(error).strip() or error.__class__.__name__
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return f"{prefix}: {message}"


@router.post(
    "/index",
    response_model=RepoIndexResponse,
    status_code=status.HTTP_201_CREATED,
)
def index_repository(
    request: RepoIndexRequest,
    indexer: Annotated[RepositoryIndexer, Depends(get_indexer)],
) -> RepoIndexResponse:
    started_at = time.perf_counter()
    try:
        result = indexer.index_url(str(request.repo_url))
    except GitError as error:
        logger.exception("Repository clone failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_error_detail(
                "Unable to clone the GitHub repository",
                error,
            ),
        ) from error
    except EmbeddingServiceError as error:
        logger.exception("Repository embedding generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_error_detail(
                "Unable to generate repository embeddings",
                error,
            ),
        ) from error
    except ValueError as error:
        logger.exception("Repository indexing validation failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Repository indexing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_detail("Repository indexing failed", error),
        ) from error

    return RepoIndexResponse(
        repo_id=result.repo_id,
        files_scanned=result.file_count,
        chunks_created=result.chunk_count,
        indexing_time_seconds=time.perf_counter() - started_at,
    )


@router.get("/{repo_id}", response_model=RepositoryMetadata)
def get_repository(
    repo_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> RepositoryMetadata:
    repository = session.get(Repository, repo_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        )

    file_count = session.scalar(
        select(func.count())
        .select_from(SourceFile)
        .where(SourceFile.repository_id == repo_id)
    )
    chunk_count = session.scalar(
        select(func.count())
        .select_from(CodeChunk)
        .where(CodeChunk.repository_id == repo_id)
    )
    return RepositoryMetadata(
        repo_id=repository.id,
        repo_url=repository.url,
        local_path=repository.local_path,
        indexed_at=repository.indexed_at,
        file_count=file_count or 0,
        chunk_count=chunk_count or 0,
    )
