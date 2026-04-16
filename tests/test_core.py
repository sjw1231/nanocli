"""Core unit tests for the nanocli agent loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from nanocli._dev.mock_server import MockResponsesServer
from nanocli.core import (
    Message,
    Session,
    ToolApprovalRequest,
    ToolCallInput,
    ToolCallOutput,
    run_turn,
)


def _output_message(*, msg_id: str, text: str) -> dict[str, Any]:
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _output_function_call(*, item_id: str, call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, separators=(",", ":")),
    }


def test_run_turn_yields_tool_call_and_approval_request_and_sends_tool_specs(tmp_path: Path) -> None:
    """Non-yolo turns must request approval before running a tool, and the API request must include tool specs."""
    session = Session(
        model="gpt-test",
        instructions="",
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        gen = run_turn(session=session, user_text="mock:single")
        assert isinstance(next(gen), Message)
        tool_call = next(gen)
        assert isinstance(tool_call, ToolCallInput)
        assert tool_call.name == "bash"
        assert "pwd" in tool_call.arguments
        approval_req = next(gen)
        assert isinstance(approval_req, ToolApprovalRequest)
        rejected = gen.send(False)
        assert isinstance(rejected, ToolCallOutput)
        assert rejected.call_id == tool_call.call_id
        assert rejected.result.success is False
        assert rejected.result.content == "rejected by user"
        try:
            next(gen)
        except StopIteration:
            pass

    tool_outputs = [item for item in session.context if isinstance(item, ToolCallOutput)]
    assert len(tool_outputs) == 1
    assert tool_outputs[0].call_id == tool_call.call_id
    assert tool_outputs[0].result.success is False
    assert tool_outputs[0].result.content == "rejected by user"

    req0 = server.requests[0]
    assert req0["store"] is False
    assert req0["stream"] is True
    assert req0["include"] == ["reasoning.encrypted_content"]
    tools = req0.get("tools")
    assert isinstance(tools, list)
    tool_names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
    assert {"bash", "read_file", "list_dir", "edit_file"} <= tool_names
    assert req0.get("tool_choice") == "auto"


def test_run_turn_yolo_executes_pwd_and_chains_with_tool_output(tmp_path: Path) -> None:
    """Runs commands without approval when yolo is enabled."""
    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="mock:single"))

    messages = [event for event in events if isinstance(event, Message)]
    outputs = [event for event in events if isinstance(event, ToolCallOutput)]
    assert len(outputs) == 1
    assert outputs[0].result.success is True
    assert "<returncode>" in outputs[0].result.content
    assert str(tmp_path) in outputs[0].result.content
    assert messages[-1].content.strip() == "pwd complete."
    assert len(server.requests) == 2
    tool_inputs = server.requests[1]["input"]
    assert any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in tool_inputs)


def test_instructions_as_system_prompt(tmp_path: Path) -> None:
    """Places instructions as the first system input when enabled."""
    session = Session(
        model="gpt-test",
        instructions="system instructions",
        instructions_as_system=True,
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer(["ok"]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        list(run_turn(session=session, user_text="hi"))

    req0 = server.requests[0]
    first_item = req0["input"][0]
    assert first_item["role"] == "system"
    assert first_item["content"] == "system instructions"
    assert "instructions" not in req0


def test_session_sends_instructions_param_when_not_system(tmp_path: Path) -> None:
    """When not using system-message instructions, we send `instructions=...` in the API request."""
    session = Session(
        model="gpt-test",
        instructions="my instructions",
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer(["ok"]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        list(run_turn(session=session, user_text="hi"))

    req0 = server.requests[0]
    assert req0.get("instructions") == "my instructions"
    first_item = req0["input"][0]
    assert first_item["role"] == "user"


def test_run_turn_tool_error_is_yielded_and_sent_to_model(tmp_path: Path) -> None:
    """Tool failures still produce `function_call_output` items so the model can react to errors."""
    response1 = {
        "output": [
            _output_message(msg_id="msg-1", text="Trying a missing file."),
            _output_function_call(
                item_id="fc-1",
                call_id="call-1",
                name="read_file",
                arguments={"path": "missing.txt", "offset": None, "limit": None},
            ),
        ]
    }
    response2 = {"output": [_output_message(msg_id="msg-2", text="done")]}

    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer([response1, response2]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="hi"))

    tool_out = [event for event in events if isinstance(event, ToolCallOutput)]
    assert tool_out
    assert tool_out[0].result.success is False

    assert len(server.requests) == 2
    tool_inputs = server.requests[1]["input"]
    assert isinstance(tool_inputs, list)
    fc_outputs = [item for item in tool_inputs if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert len(fc_outputs) == 1
    assert "not found" in fc_outputs[0]["output"]


def test_run_turn_multiple_tool_calls_in_single_response(tmp_path: Path) -> None:
    """A single model response may contain multiple tool calls; run_turn must execute all of them."""
    response1 = {
        "output": [
            _output_message(msg_id="msg-1", text="Run two commands."),
            _output_function_call(
                item_id="fc-1",
                call_id="call-1",
                name="bash",
                arguments={"command": "echo one"},
            ),
            _output_function_call(
                item_id="fc-2",
                call_id="call-2",
                name="bash",
                arguments={"command": "echo two"},
            ),
        ]
    }
    response2 = {"output": [_output_message(msg_id="msg-2", text="done")]}

    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer([response1, response2]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="hi"))

    outputs = [event for event in events if isinstance(event, ToolCallOutput)]
    assert len(outputs) == 2
    assert outputs[0].result.success is True
    assert "one" in outputs[0].result.content
    assert outputs[1].result.success is True
    assert "two" in outputs[1].result.content

    assert len(server.requests) == 2
    tool_inputs = server.requests[1]["input"]
    assert isinstance(tool_inputs, list)
    fc_outputs = [item for item in tool_inputs if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert len(fc_outputs) == 2
    call_ids = {item["call_id"] for item in fc_outputs}  # type: ignore[index]
    assert call_ids == {"call-1", "call-2"}
    for item in fc_outputs:
        assert "<stdout>" in item.get("output", "")


def test_run_turn_requests_approval_for_each_tool_call(tmp_path: Path) -> None:
    """When not yolo, approval is requested for each tool call (even if multiple calls appear in one response)."""
    response1 = {
        "output": [
            _output_message(msg_id="msg-1", text="Run two commands."),
            _output_function_call(
                item_id="fc-1",
                call_id="call-1",
                name="bash",
                arguments={"command": "echo one"},
            ),
            _output_function_call(
                item_id="fc-2",
                call_id="call-2",
                name="bash",
                arguments={"command": "echo two"},
            ),
        ]
    }
    response2 = {"output": [_output_message(msg_id="msg-2", text="done")]}

    session = Session(
        model="gpt-test",
        instructions="",
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer([response1, response2]) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        gen = run_turn(session=session, user_text="hi")

        first = next(gen)
        assert isinstance(first, Message)

        next(gen)  # tool call 1
        req1 = next(gen)
        assert isinstance(req1, ToolApprovalRequest)
        out1 = gen.send(True)
        assert isinstance(out1, ToolCallOutput)

        next(gen)  # tool call 2
        req2 = next(gen)
        assert isinstance(req2, ToolApprovalRequest)
        out2 = gen.send(True)
        assert isinstance(out2, ToolCallOutput)

        msg2 = next(gen)
        assert isinstance(msg2, Message)
        try:
            next(gen)
        except StopIteration:
            pass

    assert len(server.requests) == 2
