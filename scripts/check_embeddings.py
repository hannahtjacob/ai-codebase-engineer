from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_environment
from app.core.embedding_service import EmbeddingService, EmbeddingServiceError


def main() -> None:
    logging.disable(logging.CRITICAL)
    load_environment()
    service = EmbeddingService()
    texts = ["hello world", "def add(a, b): return a + b"]
    try:
        embeddings = service.embed_texts(texts)
    except EmbeddingServiceError as error:
        raise SystemExit(f"Embedding check failed: {error}") from None

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Dimensions: {[len(vector) for vector in embeddings]}")
    print(f"Embedding provider: {service.provider}")


if __name__ == "__main__":
    main()
