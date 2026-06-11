from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.indexer import RepositoryIndexer
from app.core.rag_engine import RagEngine


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_indexer(request: Request) -> RepositoryIndexer:
    return request.app.state.indexer


def get_rag_engine(request: Request) -> RagEngine:
    return request.app.state.rag_engine
