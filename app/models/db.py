from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.types import JSON
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    files: Mapped[list[SourceFile]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chunks: Mapped[list[CodeChunk]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    query_logs: Mapped[list[QueryLog]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SourceFile(Base):
    __tablename__ = "source_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[str] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True, nullable=False
    )
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    absolute_path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, nullable=False)

    repository: Mapped[Repository] = relationship(back_populates="files")
    chunks: Mapped[list[CodeChunk]] = relationship(
        back_populates="source_file",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CodeChunk(Base):
    __tablename__ = "code_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_id: Mapped[str] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True, nullable=False
    )
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("source_files.id", ondelete="CASCADE"), index=True, nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(Text)
    symbol_type: Mapped[str | None] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text, nullable=False)

    repository: Mapped[Repository] = relationship(back_populates="chunks")
    source_file: Mapped[SourceFile] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[str | None] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    repository: Mapped[Repository | None] = relationship(back_populates="query_logs")


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value_json: Mapped[object] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def get_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    parsed_url = make_url(url)
    engine_options: dict[str, object] = {"echo": echo}

    if parsed_url.get_backend_name() == "sqlite":
        engine_options["connect_args"] = {"check_same_thread": False}
        if parsed_url.database in {None, "", ":memory:"}:
            engine_options["poolclass"] = StaticPool
        else:
            Path(parsed_url.database).expanduser().parent.mkdir(
                parents=True, exist_ok=True
            )

    return create_engine(url, **engine_options)


def get_session_factory(
    engine: Engine | None = None,
) -> sessionmaker:
    database_engine = engine or get_engine()
    return sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )


def init_db(engine: Engine | None = None) -> Engine:
    database_engine = engine or get_engine()
    Base.metadata.create_all(database_engine)
    return database_engine
