from pathlib import Path

from app.core.file_scanner import FileScanner
from app.core.graph_builder import GraphBuilder


def test_builds_repository_import_and_call_graph(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text(
        "def format_name(name):\n"
        "    return name.title()\n",
        encoding="utf-8",
    )
    (tmp_path / "service.py").write_text(
        "from helpers import format_name as format_value\n"
        "\n"
        "def greet(name):\n"
        "    return format_value(name)\n"
        "\n"
        "def welcome(name):\n"
        "    return greet(name)\n",
        encoding="utf-8",
    )

    files = FileScanner().scan(tmp_path)
    graph = GraphBuilder().build("repo-123", files)
    data = GraphBuilder.to_data(graph)

    node_types = {node["type"] for node in data.nodes}
    assert node_types == {"Repository", "File", "Function", "Import"}

    edges = {
        (edge["source"], edge["target"], edge["type"])
        for edge in data.edges
    }
    assert (
        "repository:repo-123",
        "file:service.py",
        "CONTAINS",
    ) in edges
    assert (
        "file:service.py",
        "file:helpers.py",
        "IMPORTS",
    ) in edges
    assert (
        "function:service.py:greet",
        "function:helpers.py:format_name",
        "CALLS",
    ) in edges
    assert (
        "function:service.py:welcome",
        "function:service.py:greet",
        "CALLS",
    ) in edges
    assert (
        "file:service.py",
        "function:service.py:greet",
        "DEFINES",
    ) in edges
