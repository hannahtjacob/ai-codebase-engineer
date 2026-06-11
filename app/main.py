from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from app.api.query import router as query_router
from app.api.repos import router as repos_router
from app.core.embedding_service import EmbeddingService
from app.core.indexer import RepositoryIndexer
from app.core.rag_engine import RagEngine
from app.core.repo_loader import RepoLoader
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.models.db import get_engine, get_session_factory, init_db


def create_app(
    *,
    session_factory: sessionmaker | None = None,
    indexer: RepositoryIndexer | None = None,
    rag_engine: RagEngine | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_session_factory = session_factory
        if app_session_factory is None:
            engine = init_db(get_engine())
            app_session_factory = get_session_factory(engine)

        app.state.session_factory = app_session_factory
        if indexer is None or rag_engine is None:
            shared_embedding_service = EmbeddingService()
            shared_vector_store = VectorStore()
            app.state.indexer = indexer or RepositoryIndexer(
                app_session_factory,
                repo_loader=RepoLoader(
                    os.getenv("REPO_STORAGE_PATH", "data/repos")
                ),
                embedding_service=shared_embedding_service,
                vector_store=shared_vector_store,
            )
            app.state.rag_engine = rag_engine or RagEngine(
                retriever=Retriever(
                    embedding_service=shared_embedding_service,
                    vector_store=shared_vector_store,
                )
            )
        else:
            app.state.indexer = indexer
            app.state.rag_engine = rag_engine
        yield

    application = FastAPI(
        title="AI Codebase Engineer",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(repos_router)
    application.include_router(query_router)
    return application


app = create_app()
