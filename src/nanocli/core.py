"""Core agent loop and data model for nanocli.

This module defines the minimal data model (messages, tool outputs, tool requests)
and the agent loop that drives the Responses API tool-calling interface.
"""

import hashlib
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from nanocli.tool import ToolResult, get_tool, tool_specs


@dataclass(frozen=True)
class Message:
    """A plain chat message stored in the conversation context."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Reasoning:
    """Bridges the Responses API reasoning item into the chat context."""

    id: str
    # these fields are to conveniently bridge the Responses API's Reasoning type
    summary: list[object]
    content: list[object]
    encrypted_content: str | None


@dataclass(frozen=True)
class ToolCallInput:
    id: str | None
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCallOutput:
    """Structured output captured from executing a tool call."""

    call_id: str
    result: ToolResult


type ChatItem = Message | Reasoning | ToolCallOutput | ToolCallInput
type ChatContext = list[ChatItem]


@dataclass(frozen=True)
class ToolApprovalRequest:
    name: str
    arguments: str


type TurnEvent = ChatItem | ToolApprovalRequest

SUMMARIZATION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

SUMMARY_TEMPLATE = (
    "Another language model started to solve this problem and produced a summary of its thinking "
    "process. You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model, use the information in this summary "
    "to assist with your own analysis:\n"
    "{summary}"
)


@dataclass
class Session:
    """Holds session configuration and conversation context for a single chat."""

    model: str
    instructions: str
    instructions_as_system: bool = False
    yolo: bool = False
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    cwd: Path = field(default_factory=Path.cwd)
    context: ChatContext = field(default_factory=list)
    client: OpenAI = field(default_factory=OpenAI)
    auto_compact_token_limit: int | None = None
    last_token_usage_total_tokens: int = 0
    compaction_count: int = 0

    def __post_init__(self) -> None:
        """Derive a stable prompt cache key from the instructions."""
        digest = hashlib.sha256(self.instructions.encode("utf-8")).hexdigest()
        self.prompt_cache_key = f"nanocli:{digest}"

    def stream_response_final(self) -> tuple[list[ChatItem], int]:
        """Call the Responses API and return output items plus total-token usage."""

        def to_response_api(chat_item: ChatItem) -> ResponseInputItemParam:
            match chat_item:
                case Message(role, content):
                    return EasyInputMessageParam(role=role, content=content)
                case Reasoning(id, summary, content, encrypted_content):
                    return ResponseReasoningItemParam(
                        id=id,
                        summary=summary,  # type: ignore
                        content=content,  # type: ignore
                        encrypted_content=encrypted_content,
                        type="reasoning",
                    )
                case ToolCallOutput() as output:
                    return FunctionCallOutput(
                        call_id=output.call_id,
                        output=output.result.content,
                        type="function_call_output",
                    )
                case ToolCallInput() as input:
                    payload: ResponseFunctionToolCallParam = {
                        "call_id": input.call_id,
                        "arguments": input.arguments,
                        "name": input.name,
                        "type": "function_call",
                    }
                    if isinstance(input.id, str):
                        payload["id"] = input.id
                    return payload

        input_items: list[ResponseInputItemParam] = []
        if self.instructions_as_system:
            input_items.append(EasyInputMessageParam(role="system", content=self.instructions))
        input_items.extend(map(to_response_api, self.context))
        extra_args: dict = {}
        if self.reasoning_effort != "none":
            extra_args["reasoning"] = dict(effort=self.reasoning_effort, summary="auto")
        if not self.instructions_as_system:
            extra_args["instructions"] = self.instructions
        stream = self.client.responses.create(
            model=self.model,
            input=input_items,
            tools=tool_specs(),
            tool_choice="auto",
            include=["reasoning.encrypted_content"],
            stream=True,
            store=False,
            prompt_cache_key=self.prompt_cache_key,
            **extra_args,
        )
        # Important: don't return early from the stream loop. If we stop iterating without
        # draining the stream (or closing it), the underlying HTTP connection can stay
        # checked out and future requests may hang waiting for a free connection.
        final_outputs: list[ChatItem] | None = None
        final_total_tokens = 0
        for event in stream:
            if event.type == "response.completed":
                usage = getattr(event.response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", None)
                if isinstance(total_tokens, int):
                    final_total_tokens = total_tokens
                outputs: list[ChatItem] = []
                for output in event.response.output:
                    if output.type == "message":
                        for content in output.content:
                            if hasattr(content, "text"):
                                outputs.append(Message(output.role, content.text))  # type: ignore
                    elif output.type == "reasoning":
                        outputs.append(Reasoning(output.id, output.summary, output.content, output.encrypted_content))  # type: ignore
                    elif output.type == "function_call":
                        outputs.append(ToolCallInput(output.id, output.call_id, output.name, output.arguments))  # type: ignore
                final_outputs = outputs

        if final_outputs is not None:
            if final_total_tokens <= 0:
                raise AssertionError("Response completed without positive usage.total_tokens.")
            return final_outputs, final_total_tokens
        raise AssertionError("Stream ended without response.completed.")


def maybe_compact_session(session: Session) -> None:
    """Compact the session when the tracked token usage reaches the configured limit."""

    if session.auto_compact_token_limit is None:
        return
    if session.last_token_usage_total_tokens < session.auto_compact_token_limit:
        return

    session.context.append(Message(role="user", content=SUMMARIZATION_PROMPT))

    outputs, _ = session.stream_response_final()

    summary = ""
    for item in outputs:
        if isinstance(item, Message):
            summary = item.content.strip()
            break

    session.context = [Message(role="user", content=SUMMARY_TEMPLATE.format(summary=summary))]
    session.last_token_usage_total_tokens = 0
    session.compaction_count += 1


def run_turn(session: Session, user_text: str) -> Generator[TurnEvent, bool | None, None]:
    """Execute one user turn with the tool-calling loop."""
    # Compact before appending the new user message
    maybe_compact_session(session)

    session.context.append(Message(role="user", content=user_text))

    while True:
        maybe_compact_session(session)

        outputs, total_tokens = session.stream_response_final()
        session.last_token_usage_total_tokens = total_tokens

        tool_called = False
        for item in outputs:
            session.context.append(item)
            yield item

            if isinstance(item, ToolCallInput):
                tool = get_tool(item.name)
                if tool is None:
                    session.context.append(
                        ToolCallOutput(call_id=item.call_id, result=ToolResult(success=False, content="not found"))
                    )
                    continue

                tool_called = True
                if not tool.is_read_only() and not session.yolo:
                    approval = yield ToolApprovalRequest(name=item.name, arguments=item.arguments)
                    if not approval:
                        tool_call_output = ToolCallOutput(
                            call_id=item.call_id, result=ToolResult(success=False, content="rejected by user")
                        )
                        session.context.append(tool_call_output)
                        yield tool_call_output
                        return

                tool_result = tool.call(item.arguments, session.cwd)
                tool_call_output = ToolCallOutput(call_id=item.call_id, result=tool_result)
                session.context.append(tool_call_output)
                yield tool_call_output

        if not tool_called:
            break
