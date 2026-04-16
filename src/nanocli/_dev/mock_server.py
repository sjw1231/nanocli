import argparse
import hashlib
import http.client
import json
import math
import re
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context
from flask.typing import ResponseReturnValue
from werkzeug.serving import make_server

from nanocli.core import SUMMARIZATION_PROMPT, SUMMARY_TEMPLATE

ResponsePayload = str | dict[str, Any]

TRIGGER_TEXT_PING = "mock:text"
TRIGGER_TOOL_SINGLE_PWD = "mock:single"
TRIGGER_TOOL_SINGLE_WHOAMI = "mock:single1"
TRIGGER_TOOL_MULTI_INSPECT = "mock:multi"
TRIGGER_TOOL_EDIT_FILE = "mock:edit"
TRIGGER_TOOL_READ_FILE = "mock:read"
TRIGGER_TOOL_LONG_HORIZON = "mock:long"

SHELL_OBS_RE = re.compile(r"^Command:\n```bash\n.*?\n```\nOutput:\n", re.DOTALL)
SUMMARY_PROGRESS_RE = re.compile(r"tool_steps=(?P<steps>\d+)")
SUMMARY_TRIGGER_RE = re.compile(r"trigger=(?P<trigger>mock:[^;\s]+)")


def _build_output_message(content: str) -> dict[str, Any]:
    return {
        "id": f"msg-mock-{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }


def _build_output_function_call(*, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"fc-mock-{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": f"call-mock-{uuid.uuid4().hex}",
        "name": name,
        "arguments": json.dumps(arguments, separators=(",", ":")),
    }


def _build_response(
    request_payload: dict[str, Any],
    *,
    output_items: list[dict[str, Any]],
    response_id: str | None = None,
) -> dict[str, Any]:
    created = time.time()
    model = request_payload.get("model", "mock-model")
    input_tokens = _request_input_tokens(request_payload)
    output_tokens = _response_output_tokens(output_items)
    return {
        "id": response_id or f"resp-mock-{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created,
        "completed_at": created,
        "model": model,
        "output": output_items,
        "parallel_tool_calls": False,
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": request_payload.get("tools", []),
        "status": "completed",
        "prompt_cache_key": request_payload.get("prompt_cache_key"),
        "prompt_cache_retention": request_payload.get("prompt_cache_retention"),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _stream_completed_response(request_payload: dict[str, Any], response_payload: dict[str, Any]) -> Iterator[str]:
    yield _sse_event({"type": "response.completed", "sequence_number": 0, "response": response_payload})
    yield "data: [DONE]\n\n"


def _input_items_from_payload(request_payload: dict[str, Any]) -> list[object]:
    raw_input = request_payload.get("input")
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if isinstance(raw_input, list):
        return raw_input
    messages = request_payload.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"text", "input_text", "output_text", "reasoning_text"}:
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _serialized_token_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _approx_token_count(value)
    return _approx_token_count(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _request_input_tokens(request_payload: dict[str, Any]) -> int:
    total = 0
    for item in _input_items_from_payload(request_payload):
        total += _serialized_token_count(item)
    instructions = request_payload.get("instructions")
    if isinstance(instructions, str):
        total += _serialized_token_count(instructions)
    return total


def _response_output_tokens(output_items: list[dict[str, Any]]) -> int:
    return sum(_serialized_token_count(item) for item in output_items)


def _pick_variant(key: str, options: list[str]) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()[0]
    return options[digest % len(options)]


def _matches_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_question(text: str) -> bool:
    if "?" in text:
        return True
    words = text.split()
    if not words:
        return False
    return words[0] in {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "could",
        "should",
        "do",
        "does",
        "is",
        "are",
        "am",
        "will",
        "would",
    }


def _is_shell_observation(text: str) -> bool:
    if SHELL_OBS_RE.match(text):
        return True
    return "<stdout>" in text and "<returncode>" in text


def _is_summary_message(text: str) -> bool:
    return text.startswith(SUMMARY_TEMPLATE.partition("{summary}")[0])


def _summary_trigger(text: str) -> str | None:
    match = SUMMARY_TRIGGER_RE.search(text)
    if match is None:
        return None
    return match.group("trigger")


def _is_compaction_request(request_payload: dict[str, Any]) -> bool:
    input_items = _input_items_from_payload(request_payload)
    return _last_user_text(input_items) == SUMMARIZATION_PROMPT


def _last_user_text(input_items: list[object]) -> str:
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        text = _extract_text(item.get("content")).strip()
        if text:
            return text
    return ""


def _last_non_shell_user_text(input_items: list[object]) -> str:
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        text = _extract_text(item.get("content")).strip()
        if not text or _is_shell_observation(text) or _is_summary_message(text):
            continue
        return text
    return ""


def _find_last_trigger(input_items: list[object]) -> tuple[int, str] | None:
    trigger_set = set(PATTERN_RESPONSES.keys())
    for index in range(len(input_items) - 1, -1, -1):
        item = input_items[index]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        text = _extract_text(item.get("content")).strip()
        if not text or _is_shell_observation(text) or _is_summary_message(text) or text == SUMMARIZATION_PROMPT:
            continue
        if text in trigger_set:
            return index, text
        return None
    return None


def _find_last_summary_trigger(input_items: list[object]) -> tuple[int, str] | None:
    for index in range(len(input_items) - 1, -1, -1):
        item = input_items[index]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        text = _extract_text(item.get("content")).strip()
        if not _is_summary_message(text):
            continue
        trigger = _summary_trigger(text)
        if trigger is not None:
            return index, trigger
    return None


def _long_horizon_was_compacted(input_items: list[object]) -> bool:
    summary_match = _find_last_summary_trigger(input_items)
    if summary_match is None or summary_match[1] != TRIGGER_TOOL_LONG_HORIZON:
        return False
    trigger_match = _find_last_trigger(input_items)
    if trigger_match is None:
        return True
    return summary_match[0] > trigger_match[0]


def _count_assistant_after(input_items: list[object], start_index: int) -> int:
    count = 0
    for item in input_items[start_index + 1 :]:
        if isinstance(item, dict) and item.get("role") == "assistant":
            count += 1
    return count


def _count_function_call_outputs_after(input_items: list[object], start_index: int) -> int:
    count = 0
    for item in input_items[start_index + 1 :]:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            count += 1
    return count


def _find_last_summary_progress(input_items: list[object]) -> tuple[int, int] | None:
    for index in range(len(input_items) - 1, -1, -1):
        item = input_items[index]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        text = _extract_text(item.get("content")).strip()
        if not _is_summary_message(text):
            continue
        match = SUMMARY_PROGRESS_RE.search(text)
        if match is None:
            return None
        return index, int(match.group("steps"))
    return None


def _long_horizon_progress(input_items: list[object]) -> int:
    trigger_match = _find_last_trigger(input_items)
    summary_progress = _find_last_summary_progress(input_items)

    if trigger_match is not None:
        trigger_index, trigger = trigger_match
        if trigger != TRIGGER_TOOL_LONG_HORIZON:
            return _count_function_call_outputs_after(input_items, trigger_index)
        if summary_progress is None or trigger_index > summary_progress[0]:
            return _count_function_call_outputs_after(input_items, trigger_index)

    if summary_progress is not None:
        summary_index, completed_steps = summary_progress
        recent_outputs = _count_function_call_outputs_after(input_items, summary_index)
        return completed_steps + recent_outputs

    return 0


def _long_step_payload(label: str) -> str:
    return f"{label}-" + (label[-1] * 220)


def _long_horizon_output(step: int, *, resumed: bool) -> list[dict[str, Any]]:
    if step == 0:
        return [
            _build_output_message("Long task step 1: collecting initial data."),
            _build_output_function_call(
                name="bash",
                arguments={"command": f"echo {_long_step_payload('step-1')}"},
            ),
        ]
    if step == 1:
        return [
            _build_output_message("Long task step 2: expanding the working set."),
            _build_output_function_call(
                name="bash",
                arguments={"command": f"echo {_long_step_payload('step-2')}"},
            ),
        ]
    if step == 2:
        if resumed:
            step_text = "Long task resumed from compacted checkpoint: finishing the last action."
        else:
            step_text = "Long task step 3: finishing the last action."
        return [
            _build_output_message(step_text),
            _build_output_function_call(
                name="bash",
                arguments={"command": f"echo {_long_step_payload('step-3')}"},
            ),
        ]
    if resumed:
        return [_build_output_message("long complete after compaction.")]
    return [_build_output_message("long complete without compaction.")]


def _compaction_response_text(input_items: list[object]) -> str:
    history_items = input_items[:-1] if input_items else input_items
    trigger_match = _find_last_trigger(history_items)
    summary_trigger_match = _find_last_summary_trigger(history_items)
    if trigger_match is None:
        if summary_trigger_match is not None and summary_trigger_match[1] == TRIGGER_TOOL_LONG_HORIZON:
            progress = _long_horizon_progress(history_items)
            return (
                "Current progress: long-horizon task is underway. "
                f"trigger={TRIGGER_TOOL_LONG_HORIZON}; tool_steps={progress}; "
                "Continue from the next unfinished step."
            )
        latest = _last_non_shell_user_text(history_items)
        if latest:
            return f"Progress: latest_user={_truncate_text(latest, 160)}"
        return "Progress: no prior context."

    _, trigger = trigger_match
    if trigger == TRIGGER_TOOL_LONG_HORIZON:
        progress = _long_horizon_progress(history_items)
        return (
            "Current progress: long-horizon task is underway. "
            f"trigger={trigger}; tool_steps={progress}; "
            "Continue from the next unfinished step."
        )

    latest = _last_non_shell_user_text(history_items)
    return "Current progress: short task in progress. " f"trigger={trigger}; latest_user={_truncate_text(latest, 120)}"


def _compact_response(request_payload: dict[str, Any]) -> dict[str, Any]:
    input_items = _input_items_from_payload(request_payload)
    summary_text = _compaction_response_text(input_items)
    return _build_response(request_payload, output_items=[_build_output_message(summary_text)])


def _chatty_response(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "Mock response."
    lower = cleaned.lower()

    if lower.startswith("mock:"):
        known = ", ".join(sorted(PATTERN_RESPONSES.keys()))
        return f"Unknown mock trigger. Try one of: {known}."

    if _matches_any(lower, ("hi", "hello", "hey", "yo", "sup")):
        return _pick_variant(
            cleaned,
            [
                "Hey! I am a mock server. Ask me something.",
                "Hello! I am a mock server and ready to chat.",
                "Hi! I am a mock server. What should we try?",
            ],
        )

    if _contains_any(lower, ("thanks", "thank you", "thx")):
        return _pick_variant(cleaned, ["You are welcome.", "Anytime.", "Glad to help."])

    if _contains_any(lower, ("bye", "goodbye", "see ya", "later")):
        return _pick_variant(cleaned, ["Later!", "Bye for now.", "See you soon."])

    if lower in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
        return _pick_variant(cleaned, ["Got it.", "Sounds good.", "Okay."])

    if lower in {"no", "nope", "nah"}:
        return _pick_variant(cleaned, ["Understood.", "Okay, no problem.", "All right."])

    if _contains_any(lower, ("help", "what can you do", "commands")):
        known = ", ".join(sorted(PATTERN_RESPONSES.keys()))
        return f"I am a mock Responses API server. Try one of these triggers: {known}."

    if _contains_any(lower, ("error", "bug", "failed", "traceback")):
        return "That sounds annoying. If you paste the exact error text, I can respond with a mock diagnosis."

    if _looks_like_question(lower):
        topic = _truncate_text(cleaned, 160).rstrip("?")
        return (
            "Good question. I do not have real data here, but a quick approach is to "
            f"start small and verify assumptions: {topic}."
        )

    if len(cleaned.split()) <= 3:
        return f"Noted: {_truncate_text(cleaned, 120)}."

    return f"Got it. You said: {_truncate_text(cleaned, 160)}."


def _fallback_response_text(request_payload: dict[str, Any]) -> str:
    messages = _input_items_from_payload(request_payload)
    text = _last_non_shell_user_text(messages)
    if text:
        return _chatty_response(text)
    return "Mock response."


PATTERN_RESPONSES: dict[str, list[list[dict[str, Any]]]] = {
    TRIGGER_TEXT_PING: [[_build_output_message("pong")]],
    TRIGGER_TOOL_SINGLE_PWD: [
        [
            _build_output_message("I'll check the working directory:\n```bash\npwd\n```"),
            _build_output_function_call(name="bash", arguments={"command": "pwd"}),
        ],
        [_build_output_message("pwd complete.")],
    ],
    TRIGGER_TOOL_SINGLE_WHOAMI: [
        [
            _build_output_message("Let's run `whoami`:\n```bash\nwhoami\n```"),
            _build_output_function_call(name="bash", arguments={"command": "whoami"}),
        ],
        [_build_output_message("whoami complete.")],
    ],
    TRIGGER_TOOL_MULTI_INSPECT: [
        [
            _build_output_message("Let's check the current directory:\n```bash\npwd\n```"),
            _build_output_function_call(name="bash", arguments={"command": "pwd"}),
        ],
        [
            _build_output_message("I will run `ls` to confirm:\n```bash\nls\n```"),
            _build_output_function_call(name="list_dir", arguments={"path": "."}),
        ],
        [_build_output_message("inspect complete.")],
    ],
    TRIGGER_TOOL_READ_FILE: [
        [
            _build_output_message("Creating a temp marker file."),
            _build_output_function_call(
                name="bash",
                arguments={
                    "command": "mkdir -p .nanocli_mock && printf 'hello\\n' > .nanocli_mock/marker.txt",
                },
            ),
        ],
        [
            _build_output_message("Reading the temp marker file."),
            _build_output_function_call(
                name="read_file",
                arguments={"path": ".nanocli_mock/marker.txt", "offset": 1, "limit": 2000},
            ),
        ],
        [_build_output_message("read complete.")],
    ],
    TRIGGER_TOOL_EDIT_FILE: [
        [
            _build_output_message("Creating a temp marker file."),
            _build_output_function_call(
                name="bash",
                arguments={
                    "command": "mkdir -p .nanocli_mock && printf 'hello\\n' > .nanocli_mock/marker.txt",
                },
            ),
        ],
        [
            _build_output_message("Editing the temp marker file."),
            _build_output_function_call(
                name="edit_file",
                arguments={
                    "path": ".nanocli_mock/marker.txt",
                    "search": "hello",
                    "replace": "goodbye",
                },
            ),
        ],
        [_build_output_message("edit complete.")],
    ],
    TRIGGER_TOOL_LONG_HORIZON: [[_build_output_message("long complete.")]],
}


class _ServerState:
    def __init__(
        self,
        responses: Iterable[ResponsePayload],
        default_response: ResponsePayload | None = None,
    ) -> None:
        self.responses: deque[ResponsePayload] = deque(responses)
        self.default_response = default_response
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def record_request(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(payload)

    def next_response(self, request_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        pattern = self._pattern_response(request_payload)
        if pattern is not None:
            return 200, pattern

        with self.lock:
            if self.responses:
                response = self.responses.popleft()
            else:
                response = None

        if response is not None:
            return 200, _coerce_response_payload(request_payload, response)

        if _is_compaction_request(request_payload):
            return 200, _compact_response(request_payload)

        if self.default_response is not None:
            return 200, _coerce_response_payload(request_payload, self.default_response)

        return 200, _build_response(
            request_payload, output_items=[_build_output_message(_fallback_response_text(request_payload))]
        )

    def _pattern_response(self, request_payload: dict[str, Any]) -> dict[str, Any] | None:
        if _is_compaction_request(request_payload):
            return None
        input_items = _input_items_from_payload(request_payload)
        match = _find_last_trigger(input_items)
        if match is None:
            summary_match = _find_last_summary_trigger(input_items)
            if summary_match is None or summary_match[1] != TRIGGER_TOOL_LONG_HORIZON:
                return None
            output_items = _long_horizon_output(
                _long_horizon_progress(input_items),
                resumed=_long_horizon_was_compacted(input_items),
            )
            return _build_response(
                request_payload,
                output_items=output_items,
                response_id=f"resp-mock-{uuid.uuid4().hex}",
            )
        trigger_index, trigger = match
        if trigger == TRIGGER_TOOL_LONG_HORIZON:
            output_items = _long_horizon_output(
                _long_horizon_progress(input_items),
                resumed=_long_horizon_was_compacted(input_items),
            )
        else:
            step = max(
                _count_assistant_after(input_items, trigger_index),
                _count_function_call_outputs_after(input_items, trigger_index),
            )
            steps = PATTERN_RESPONSES[trigger]
            output_items = steps[step] if step < len(steps) else steps[-1]
        return _build_response(
            request_payload,
            output_items=output_items,
            response_id=f"resp-mock-{uuid.uuid4().hex}",
        )


def _coerce_response_payload(request_payload: dict[str, Any], response: ResponsePayload) -> dict[str, Any]:
    if isinstance(response, dict):
        if response.get("object") == "response":
            return response
        output_items = response.get("output")
        if isinstance(output_items, list):
            return _build_response(request_payload, output_items=output_items)
        return _build_response(request_payload, output_items=[_build_output_message(json.dumps(response))])
    return _build_response(request_payload, output_items=[_build_output_message(str(response))])


def _create_app(state: _ServerState) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> ResponseReturnValue:
        return jsonify({"status": "ok"}), 200

    @app.post("/v1/responses")
    def responses() -> ResponseReturnValue:
        payload = request.get_json(silent=True) or {}
        state.record_request(payload)
        status, response_payload = state.next_response(payload)
        if payload.get("stream"):
            return Response(
                stream_with_context(_stream_completed_response(payload, response_payload)),
                status=status,
                mimetype="text/event-stream",
            )
        return jsonify(response_payload), status

    @app.post("/__shutdown__")
    def shutdown() -> ResponseReturnValue:
        shutdown_fn = request.environ.get("werkzeug.server.shutdown")
        if shutdown_fn is None:
            return jsonify({"error": {"message": "Shutdown not supported."}}), 500
        shutdown_fn()
        return jsonify({"status": "shutting down"}), 200

    return app


class MockResponsesServer:
    def __init__(
        self,
        responses: Iterable[ResponsePayload] | None = None,
        *,
        default_response: ResponsePayload | None = None,
    ) -> None:
        self._state = _ServerState(responses or [], default_response=default_response)
        self._app = _create_app(self._state)
        self._thread: threading.Thread | None = None
        self._server: Any | None = None
        self.base_url: str | None = None
        self._host: str | None = None
        self._port: int | None = None

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self._state.requests

    @property
    def app(self) -> Flask:
        return self._app

    def push(self, response: ResponsePayload) -> None:
        with self._state.lock:
            self._state.responses.append(response)

    def _serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("server not initialized")
        self._server.serve_forever(poll_interval=0.05)

    def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        if self._thread is not None:
            return
        if port == 0:
            port = _find_free_port(host)
        self._host = host
        self._port = port
        self.base_url = f"http://{host}:{port}/v1"
        self._server = make_server(host, port, self._app, threaded=True)
        self._thread = threading.Thread(
            target=self._serve_forever,
            name="mock-responses-server",
            daemon=True,
        )
        self._thread.start()
        if not _wait_for_server(host, port, timeout=1.0):
            raise RuntimeError("Mock server failed to start.")

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._app.run(host=host, port=port, threaded=True, use_reloader=False)

    def stop(self) -> None:
        if self._thread is None or self._server is None:
            return
        self._server.shutdown()
        self._thread.join(timeout=1)
        self._server.server_close()
        self._server = None
        self._thread = None
        self._host = None
        self._port = None

    def __enter__(self) -> "MockResponsesServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()


MockChatCompletionServer = MockResponsesServer


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=0.2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return True
        except OSError:
            time.sleep(0.02)
    return False


def _load_responses(path: Path) -> list[ResponsePayload]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanocli-mock")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--response",
        action="append",
        default=[],
        help="Raw response payload (string assistant message or JSON object). Can be repeated.",
    )
    parser.add_argument(
        "--responses-file",
        type=Path,
        default=None,
        help="Path to a JSON list of response payloads.",
    )
    parser.add_argument(
        "--default-response",
        default=None,
        help="Fallback assistant response when the queue is empty.",
    )
    args = parser.parse_args(argv)

    responses: list[ResponsePayload] = []
    for raw in args.response:
        if raw.strip().startswith("{") or raw.strip().startswith("["):
            responses.append(json.loads(raw))
        else:
            responses.append(raw)

    if args.responses_file is not None:
        if not args.responses_file.exists():
            parser.error(f"Responses file not found: {args.responses_file}")
        responses.extend(_load_responses(args.responses_file))

    default_response: ResponsePayload | None = args.default_response
    server = MockResponsesServer(responses, default_response=default_response)
    print(f"Mock server listening on http://{args.host}:{args.port}/v1")
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
