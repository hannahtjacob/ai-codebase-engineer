from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from git.exc import GitError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_indexer, get_session
from app.core.embedding_service import EmbeddingServiceError
from app.core.indexer import RepositoryIndexer
from app.models.db import CodeChunk, Repository, SourceFile
from app.models.schemas import (
    RepoIndexRequest,
    RepoIndexResponse,
    RepositoryMetadata,
)


router = APIRouter(prefix="/repos", tags=["repositories"])


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
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to clone the GitHub repository.",
        ) from error
    except EmbeddingServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to generate repository embeddings.",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Repository indexing failed.",
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
