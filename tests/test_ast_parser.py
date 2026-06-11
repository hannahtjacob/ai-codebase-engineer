from app.core.ast_parser import PythonASTParser


def test_extracts_imports_functions_classes_methods_and_calls() -> None:
    parsed = PythonASTParser().parse(
        "from helpers import format_name\n"
        "import os\n"
        "\n"
        "def greet(name):\n"
        "    return format_name(name)\n"
        "\n"
        "class Greeter:\n"
        "    def run(self, name):\n"
        "        return greet(name)\n",
        "app/service.py",
    )

    assert [
        (node.node_type, node.name, node.line)
        for node in parsed.nodes
    ] == [
        ("File", "app/service.py", None),
        ("Import", "helpers.format_name", 1),
        ("Import", "os", 2),
        ("Function", "greet", 4),
        ("Class", "Greeter", 7),
        ("Function", "Greeter.run", 8),
    ]
    assert [(item.module, item.imported_name, item.line) for item in parsed.imports] == [
        ("helpers", "format_name", 1),
        ("os", None, 2),
    ]
    assert [(call.name, call.line) for call in parsed.calls] == [
        ("format_name", 5),
        ("greet", 9),
    ]
    assert {edge.edge_type for edge in parsed.edges} == {"DEFINES", "CONTAINS"}
