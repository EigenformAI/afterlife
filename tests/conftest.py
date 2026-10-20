import json
from pathlib import Path

import pytest

from afterlife.cli import main


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / ".afterlife"
    monkeypatch.setenv("AFTERLIFE_HOME", str(root))
    assert main(["init", "--json"]) == 0
    return root


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]):
    """Run the CLI with --json; return (exit_code, parsed_output)."""

    def run(*argv: str) -> tuple[int, dict]:
        capsys.readouterr()
        code = main([*argv, "--json"])
        return code, json.loads(capsys.readouterr().out)

    return run


def experience(text: str, *, tags=(), priority=None, session="s1", **extra) -> dict:
    exp = {
        "messages": [
            {"role": "user", "content": f"prompt for {text}"},
            {"role": "assistant", "content": f"answer about {text}"},
        ],
        "tags": list(tags),
        "session_id": session,
        **extra,
    }
    if priority is not None:
        exp["priority"] = priority
    return exp


@pytest.fixture
def write_json(tmp_path: Path):
    counter = iter(range(10**6))

    def write(payload) -> str:
        path = tmp_path / f"payload-{next(counter)}.json"
        if isinstance(payload, list):
            path.write_text("\n".join(json.dumps(p) for p in payload))
        else:
            path.write_text(json.dumps(payload))
        return str(path)

    return write
