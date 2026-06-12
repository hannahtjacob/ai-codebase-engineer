from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_rag_engine, get_session
from app.core.embedding_service import EmbeddingServiceError
from app.core.rag_engine import (
    MissingOpenAIAPIKeyError,
    OllamaUnavailableError,
    RagEngine,
    RagEngineError,
)
from app.models.db import QueryLog, Repository
from app.models.schemas import (
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)


router = APIRouter(tags=["queries"])
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
def query_repository(
    request: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
    rag_engine: Annotated[RagEngine, Depends(get_rag_engine)],
) -> QueryResponse:
    if session.get(Repository, request.repo_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        )

    started_at = time.perf_counter()
    try:
        result = rag_engine.answer_question(
            request.repo_id,
            request.question,
            top_k=request.top_k,
        )
    except MissingOpenAIAPIKeyError as error:
        logger.exception("OpenAI configuration error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except OllamaUnavailableError as error:
        logger.exception("Ollama is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except RagEngineError as error:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    except (EmbeddingServiceError, RuntimeError) as error:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed.",
        ) from error

    duration_ms = (time.perf_counter() - started_at) * 1000
    session.add(
        QueryLog(
            repository_id=request.repo_id,
            query=request.question,
            response=str(result["answer"]),
            duration_ms=duration_ms,
        )
    )
    try:
        session.commit()
    except Exception as error:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The answer was generated but query history could not be saved.",
        ) from error

    return QueryResponse(
        answer=str(result["answer"]),
        sources=[
            SourceCitation.model_validate(source)
            for source in result["sources"]
        ],
    )


@router.get(
    "/query/history/{repo_id}",
    response_model=list[QueryHistoryItem],
)
def get_query_history(
    repo_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> list[QueryHistoryItem]:
    if session.get(Repository, repo_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        )

    logs = session.scalars(
        select(QueryLog)
        .where(QueryLog.repository_id == repo_id)
        .order_by(QueryLog.created_at.desc(), QueryLog.id.desc())
    ).all()
    return [
        QueryHistoryItem(
            id=log.id,
            repo_id=log.repository_id,
            question=log.query,
            answer=log.response,
            duration_ms=log.duration_ms,
            created_at=log.created_at,
        )
        for log in logs
    ]
