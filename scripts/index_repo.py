from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.indexer import RepositoryIndexer
from app.core.repo_loader import RepoLoader
from app.models.db import get_engine, get_session_factory, init_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone, scan, chunk, and persist a source repository."
    )
    parser.add_argument("repo_url", help="Git repository URL to index")
    args = parser.parse_args()

    engine = get_engine()
    init_db(engine)
    indexer = RepositoryIndexer(
        get_session_factory(engine),
        repo_loader=RepoLoader(os.getenv("REPO_STORAGE_PATH", "data/repos")),
    )
    result = indexer.index_url(args.repo_url)

    print(
        f"Indexed {result.file_count} files and {result.chunk_count} chunks "
        f"for repository {result.repo_id}"
    )


if __name__ == "__main__":
    main()
