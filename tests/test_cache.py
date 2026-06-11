from datetime import datetime, timedelta, timezone

from sqlalchemy import inspect

from app.core.cache import SQLiteCache
from app.models.db import get_engine


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def test_cache_hit_miss_and_json_round_trip() -> None:
    engine = get_engine("sqlite:///:memory:")
    cache = SQLiteCache(engine)

    assert cache.get("missing") is None

    value = {"answer": "hello", "scores": [0.1, 0.2], "valid": True}
    cache.set("key", value)

    assert cache.get("key") == value
    assert "cache_entries" in inspect(engine).get_table_names()


def test_cache_delete() -> None:
    cache = SQLiteCache(database_url="sqlite:///:memory:")
    cache.set("key", {"value": 1})

    cache.delete("key")

    assert cache.get("key") is None


def test_cache_entry_expires() -> None:
    clock = Clock()
    cache = SQLiteCache(
        database_url="sqlite:///:memory:",
        clock=clock,
    )
    cache.set("key", ["cached"], ttl_seconds=60)

    clock.now += timedelta(seconds=59)
    assert cache.get("key") == ["cached"]

    clock.now += timedelta(seconds=1)
    assert cache.get("key") is None
