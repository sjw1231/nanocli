"""Compaction-focused tests for HW4."""

from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI

from nanocli._dev.mock_server import MockResponsesServer
from nanocli.core import (
    SUMMARIZATION_PROMPT,
    SUMMARY_TEMPLATE,
    Message,
    Session,
    ToolCallOutput,
    maybe_compact_session,
    run_turn,
)


def _request_user_messages(request_payload: dict[str, object]) -> list[str]:
    messages: list[str] = []
    for item in request_payload["input"]:  # type: ignore[index]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            messages.append(content)
    return messages


def _make_session(tmp_path: Path, **kwargs: Any) -> Session:
    return Session(
        model="gpt-test",
        instructions="",
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        **kwargs,
    )


def test_maybe_compact_session_returns_early_without_limit(tmp_path: Path) -> None:
    """Compaction is disabled when no automatic limit is configured."""
    no_limit = _make_session(tmp_path)
    no_limit.context = [Message(role="user", content="keep this context")]
    no_limit.last_token_usage_total_tokens = 500
    maybe_compact_session(no_limit)
    assert no_limit.context == [Message(role="user", content="keep this context")]
    assert no_limit.compaction_count == 0


def test_maybe_compact_session_returns_early_below_limit(tmp_path: Path) -> None:
    """Compaction should not run until tracked usage reaches the configured limit."""
    no_usage = _make_session(tmp_path, auto_compact_token_limit=100)
    no_usage.context = [Message(role="user", content="keep this context")]
    maybe_compact_session(no_usage)
    assert no_usage.context == [Message(role="user", content="keep this context")]
    assert no_usage.compaction_count == 0

    below_limit = _make_session(tmp_path, auto_compact_token_limit=700)
    below_limit.context = [Message(role="user", content="keep this context")]
    below_limit.last_token_usage_total_tokens = 699
    maybe_compact_session(below_limit)
    assert below_limit.context == [Message(role="user", content="keep this context")]
    assert below_limit.compaction_count == 0

    baseline_only = _make_session(tmp_path, auto_compact_token_limit=100)
    baseline_only.last_token_usage_total_tokens = 99
    baseline_only.context = [Message(role="user", content="x" * 10_000)]
    maybe_compact_session(baseline_only)
    assert baseline_only.context == [Message(role="user", content="x" * 10_000)]
    assert baseline_only.compaction_count == 0


def test_maybe_compact_session_uses_fallback_prompt_as_synthetic_user_message(tmp_path: Path) -> None:
    """Compaction should append the fixed fallback prompt and store the checkpoint template."""
    session = _make_session(tmp_path)
    session.context = [
        Message(role="user", content="Investigate the repository state."),
        Message(role="assistant", content="I started exploring."),
    ]
    session.last_token_usage_total_tokens = 100
    session.auto_compact_token_limit = 100

    with MockResponsesServer(["summary"]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        maybe_compact_session(session)

    assert session.context == [Message(role="user", content=SUMMARY_TEMPLATE.format(summary="summary"))]
    compact_request = server.requests[0]
    assert _request_user_messages(compact_request)[-1] == SUMMARIZATION_PROMPT


def test_maybe_compact_session_uses_fallback_when_summary_is_blank(tmp_path: Path) -> None:
    """Blank compaction summaries still produce a checkpoint message with the fixed template."""
    session = _make_session(tmp_path)
    session.context = [Message(role="user", content="Investigate the repository state.")]
    session.last_token_usage_total_tokens = 100
    session.auto_compact_token_limit = 100

    with MockResponsesServer(["   "]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        maybe_compact_session(session)

    assert session.context == [Message(role="user", content=SUMMARY_TEMPLATE.format(summary=""))]


def test_maybe_compact_session_clears_context(tmp_path: Path) -> None:
    """A completed compaction replaces old history and resets tracked state."""
    session = _make_session(tmp_path)
    session.context = [
        Message(role="user", content="first"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="second"),
    ]
    session.last_token_usage_total_tokens = 123
    session.auto_compact_token_limit = 123

    with MockResponsesServer(["summary"]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        maybe_compact_session(session)

    assert session.context == [Message(role="user", content=SUMMARY_TEMPLATE.format(summary="summary"))]
    assert session.last_token_usage_total_tokens == 0
    assert session.compaction_count == 1


def test_run_turn_compacts_before_recording_new_user_message(tmp_path: Path) -> None:
    """A new turn should compact first so the current user request is preserved afterward."""
    session = _make_session(tmp_path)
    session.context = [Message(role="user", content="old request")]
    session.last_token_usage_total_tokens = 100
    session.auto_compact_token_limit = 100

    with MockResponsesServer(["summary", "done"]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="new request"))

    compact_request_user_messages = _request_user_messages(server.requests[0])
    follow_up_request_user_messages = _request_user_messages(server.requests[1])

    assert compact_request_user_messages[-1] == SUMMARIZATION_PROMPT
    assert "new request" not in compact_request_user_messages
    assert follow_up_request_user_messages == [SUMMARY_TEMPLATE.format(summary="summary"), "new request"]
    assert [event.content for event in events if isinstance(event, Message)] == ["done"]


def test_stream_response_final_returns_total_token_usage(tmp_path: Path) -> None:
    """The response bridge should return the completed response's total token usage."""
    session = _make_session(tmp_path, yolo=True)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        session.context.append(Message(role="user", content="mock:text"))
        _, total_tokens = session.stream_response_final()

    assert isinstance(total_tokens, int)
    assert total_tokens > 0


def test_stream_response_final_requires_positive_total_token_usage(tmp_path: Path) -> None:
    """Responses without positive total-token usage are rejected as invalid compaction baselines."""
    session = _make_session(tmp_path, yolo=True)
    response = {
        "object": "response",
        "id": "resp-mock-zero-usage",
        "created_at": 0,
        "completed_at": 0,
        "model": "gpt-test",
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 0},
    }

    with MockResponsesServer([response]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        session.context.append(Message(role="user", content="hi"))
        with pytest.raises(AssertionError):
            session.stream_response_final()


def test_retained_mock_commands_still_work(tmp_path: Path) -> None:
    """Retained mock commands should still behave as before, and non-exact mentions should not trigger them."""
    cases = [
        ("mock:text", "pong", 0),
        ("mock:single1", "whoami complete.", 1),
        ("mock:multi", "inspect complete.", 2),
        ("mock:read", "read complete.", 2),
        ("mock:edit", "edit complete.", 2),
    ]

    for trigger, final_message, tool_outputs in cases:
        session = _make_session(tmp_path, yolo=True)
        with MockResponsesServer() as server:
            session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
            events = list(run_turn(session=session, user_text=trigger))
        messages = [event.content for event in events if isinstance(event, Message)]
        outputs = [event for event in events if isinstance(event, ToolCallOutput)]
        assert messages[-1] == final_message
        assert len(outputs) == tool_outputs
        assert all(output.result.success for output in outputs)

    session = _make_session(tmp_path, yolo=True)
    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="please explain mock:text"))
    messages = [event.content for event in events if isinstance(event, Message)]
    assert messages == ["Noted: please explain mock:text."]


def test_run_turn_does_not_compact_when_threshold_not_reached(tmp_path: Path) -> None:
    """Long tasks should proceed normally when the compaction threshold is set high enough."""
    session = _make_session(tmp_path, yolo=True, auto_compact_token_limit=10_000)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        list(run_turn(session=session, user_text="mock:long"))

    assert session.compaction_count == 0
    assert all(_request_user_messages(request)[-1] != SUMMARIZATION_PROMPT for request in server.requests)


def test_run_turn_auto_compacts_long_running_task_with_mock_server(tmp_path: Path) -> None:
    """The long-horizon mock flow should compact and then resume to completion."""
    session = _make_session(tmp_path, yolo=True, auto_compact_token_limit=320)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="mock:long"))

    assistant_messages = [event.content for event in events if isinstance(event, Message)]
    tool_outputs = [event for event in events if isinstance(event, ToolCallOutput)]

    assert assistant_messages[-1] == "long complete after compaction."
    assert len(tool_outputs) == 3
    assert session.compaction_count >= 1


def test_second_explicit_mock_long_starts_a_fresh_long_task(tmp_path: Path) -> None:
    """A second explicit long-task request should start over instead of resuming old compacted state."""
    session = _make_session(tmp_path, yolo=True, auto_compact_token_limit=320)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        list(run_turn(session=session, user_text="mock:long"))
        second_events = list(run_turn(session=session, user_text="mock:long"))

    assistant_messages = [event.content for event in second_events if isinstance(event, Message)]
    assert assistant_messages[0] == "Long task step 1: collecting initial data."
    assert assistant_messages[0] != "long complete after compaction."


def test_run_turn_follow_up_request_contains_only_summary_user_message(tmp_path: Path) -> None:
    """The post-compaction follow-up request should carry only the checkpoint user message."""
    session = _make_session(tmp_path, yolo=True, auto_compact_token_limit=320)

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        list(run_turn(session=session, user_text="mock:long"))

    compact_request_indexes = [
        index
        for index, request in enumerate(server.requests)
        if _request_user_messages(request)[-1] == SUMMARIZATION_PROMPT
    ]
    assert compact_request_indexes
    follow_up_request = server.requests[compact_request_indexes[0] + 1]
    follow_up_user_messages = _request_user_messages(follow_up_request)
    assert len(follow_up_user_messages) == 1
    assert follow_up_user_messages[0].startswith(SUMMARY_TEMPLATE.partition("{summary}")[0])
    assert "trigger=mock:long" in follow_up_user_messages[0]
    assert "tool_steps=" in follow_up_user_messages[0]
