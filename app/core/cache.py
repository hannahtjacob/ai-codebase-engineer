from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.models.db import CacheEntry, get_engine


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteCache:
    def __init__(
        self,
        engine: Engine | None = None,
        *,
        database_url: str | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if engine is not None and database_url is not None:
            raise ValueError("provide either engine or database_url, not both")

        self.engine = engine or get_engine(database_url)
        CacheEntry.__table__.create(self.engine, checkfirst=True)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self._clock = clock

    def get(self, key: str) -> Any | None:
        self._validate_key(key)
        now = self._normalize_datetime(self._clock())

        with self.session_factory.begin() as session:
            entry = session.get(CacheEntry, key)
            if entry is None:
                return None
            expires_at = self._normalize_datetime(entry.expires_at)
            if expires_at is not None and expires_at <= now:
                session.delete(entry)
                return None
            return entry.value_json

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
    ) -> None:
        self._validate_key(key)
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")

        created_at = self._normalize_datetime(self._clock())
        assert created_at is not None
        expires_at = (
            created_at + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )

        with self.session_factory.begin() as session:
            entry = session.get(CacheEntry, key)
            if entry is None:
                session.add(
                    CacheEntry(
                        key=key,
                        value_json=value,
                        created_at=created_at,
                        expires_at=expires_at,
                    )
                )
            else:
                entry.value_json = value
                entry.created_at = created_at
                entry.expires_at = expires_at

    def delete(self, key: str) -> None:
        self._validate_key(key)
        with self.session_factory.begin() as session:
            session.execute(delete(CacheEntry).where(CacheEntry.key == key))

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        if not key:
            raise ValueError("key must not be empty")

    @staticmethod
    def _normalize_datetime(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


Cache = SQLiteCache
