from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_graph_builder, get_session
from app.core.graph_builder import GraphBuilder
from app.models.db import Repository
from app.models.schemas import GraphResponse


router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/{repo_id}", response_model=GraphResponse)
def get_repository_graph(
    repo_id: str,
    session: Annotated[Session, Depends(get_session)],
    graph_builder: Annotated[GraphBuilder, Depends(get_graph_builder)],
) -> GraphResponse:
    repository = session.get(Repository, repo_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        )

    repository_path = Path(repository.local_path)
    if not repository_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The indexed repository checkout is no longer available.",
        )

    try:
        graph = graph_builder.build_from_path(repo_id, repository_path)
        data = graph_builder.to_data(graph)
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to build the repository graph.",
        ) from error

    return GraphResponse(nodes=data.nodes, edges=data.edges)
