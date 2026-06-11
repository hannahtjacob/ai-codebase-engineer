import json
from pathlib import Path

import pytest

from scripts.run_eval import load_cases, save_report


def test_load_cases_reads_jsonl(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "repo_url": "https://github.com/example/repo",
                "question": "Where is auth?",
                "expected_files": ["./app/auth.py"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [case] = load_cases(questions_path)

    assert case.repo_url == "https://github.com/example/repo"
    assert case.expected_files == ("app/auth.py",)


def test_load_cases_reports_invalid_line_number(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text("\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"questions\.jsonl:2"):
        load_cases(questions_path)


def test_save_report_creates_json_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "results.json"

    save_report(output_path, {"summary": {"recall_at_5": 1.0}})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "summary": {"recall_at_5": 1.0}
    }
