"""Terminal UI tests for nanocli."""

from io import StringIO
from pathlib import Path
from typing import Any

from openai import OpenAI

from nanocli._dev.mock_server import MockResponsesServer
from nanocli.cli import run_and_print
from nanocli.core import Session


def _make_session(tmp_path: Path, **kwargs: Any) -> Session:
    return Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        **kwargs,
    )


def test_run_and_print_shows_compaction_notice(tmp_path: Path) -> None:
    session = _make_session(tmp_path, auto_compact_token_limit=320)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        output = StringIO()
        run_and_print(session, "mock:long", stream=output)

    rendered = output.getvalue()
    assert "Notice" in rendered
    assert "Context compacted. Continuing from checkpoint." in rendered
    assert "long complete after compaction." in rendered


def test_run_and_print_omits_compaction_notice_when_not_needed(tmp_path: Path) -> None:
    session = _make_session(tmp_path)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        output = StringIO()
        run_and_print(session, "mock:text", stream=output)

    rendered = output.getvalue()
    assert "Context compacted. Continuing from checkpoint." not in rendered
