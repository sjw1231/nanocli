"""Terminal UI for nanocli.

The CLI is intentionally thin: it formats output, requests approval for tool
calls, and delegates all agent behavior to core.run_turn.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import TextIO

from nanocli.core import (
    SUMMARIZATION_PROMPT,
    Message,
    Reasoning,
    Session,
    ToolApprovalRequest,
    ToolCallInput,
    ToolCallOutput,
    run_turn,
)

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_COLORS = {
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
}
LABEL_STYLES = {
    "assistant": ANSI_BOLD + ANSI_COLORS["green"],
    "reasoning": ANSI_BOLD + ANSI_COLORS["cyan"],
    "tool call": ANSI_BOLD + ANSI_COLORS["yellow"],
    "tool result": ANSI_BOLD + ANSI_COLORS["magenta"],
    "notice": ANSI_BOLD + ANSI_COLORS["cyan"],
    "error": ANSI_COLORS["red"],
}
TEXT_STYLES = {
    "assistant": ANSI_COLORS["green"],
    "reasoning": ANSI_COLORS["cyan"],
    "tool call": ANSI_COLORS["yellow"],
    "tool result": ANSI_COLORS["magenta"],
    "notice": ANSI_COLORS["cyan"],
    "error": ANSI_COLORS["red"],
}


def _style_label(label: str, stream: TextIO) -> str:
    """Apply terminal styling to a label if stdout is a TTY."""
    if not getattr(stream, "isatty", lambda: False)() or os.environ.get("NO_COLOR"):
        return label
    style = LABEL_STYLES.get(label.lower())
    if not style:
        return label
    return f"{style}{label}{ANSI_RESET}"


def _style_text(text: str, style_key: str | None, stream: TextIO) -> str:
    """Apply terminal styling to text if stdout is a TTY."""
    if style_key is None:
        return text
    if not getattr(stream, "isatty", lambda: False)() or os.environ.get("NO_COLOR"):
        return text
    style = TEXT_STYLES.get(style_key)
    if not style:
        return text
    return f"{style}{text}{ANSI_RESET}"


def print_text(text: str, style_key: str | None = None, stream: TextIO = sys.stdout) -> None:
    """Print possibly-multi-line text with optional color styling."""
    lines = text.splitlines() or [""]
    for line in lines:
        print(_style_text(line, style_key=style_key, stream=stream), file=stream, flush=True)


def print_labeled(label: str, text: str | None, stream: TextIO = sys.stdout) -> None:
    """Print a label and its associated text block."""
    print(_style_label(label, stream=stream), file=stream, flush=True)
    if text is None:
        return
    print_text(text, style_key=label.lower(), stream=stream)


def format_prompt(label: str, stream: TextIO = sys.stdout) -> str:
    """Format the interactive prompt with optional styling."""
    if not getattr(stream, "isatty", lambda: False)() or os.environ.get("NO_COLOR"):
        return f"{label}> "
    return f"{ANSI_BOLD}{label}>{ANSI_RESET} "


def run_and_print(session: Session, user_text: str, stream: TextIO = sys.stdout) -> None:
    """Run one agent turn and print events to the terminal."""
    gen = run_turn(session=session, user_text=user_text)
    approval: bool | None = None
    seen_compaction_count = session.compaction_count
    while True:
        try:
            event = gen.send(approval) if approval is not None else next(gen)
        except StopIteration:
            return
        if session.compaction_count > seen_compaction_count:
            print_labeled("Notice", "Context compacted. Continuing from checkpoint.", stream=stream)
            seen_compaction_count = session.compaction_count
        approval = None
        if isinstance(event, ToolApprovalRequest):
            assert not session.yolo
            response = input(f"Run {event.name} {event.arguments}? [y/N] ").strip().lower()
            approval = response in {"y", "yes"}
            continue
        if isinstance(event, Message) and event.role == "assistant":
            print_labeled("Assistant", event.content, stream=stream)
        elif isinstance(event, ToolCallInput):
            print_labeled("Tool call", f"{event.name} {event.arguments}", stream=stream)
        elif isinstance(event, ToolCallOutput):
            if event.result.success:
                print_labeled("Tool result", event.result.content, stream=stream)
            else:
                print_labeled("Tool result", f"<error>\n{event.result.content}\n</error>", stream=stream)
        elif isinstance(event, Reasoning):
            reasoning = _extract_reasoning_text(event)
            if reasoning.strip():
                print_labeled("Reasoning", reasoning, stream=stream)


def _extract_reasoning_text(event: Reasoning) -> str:
    """Extract display text from a reasoning event."""
    content = event.content or []
    summary = event.summary or []
    lines: list[str] = []
    for obj in content + summary:
        if isinstance(obj, dict):
            text = obj.get("text")
            if isinstance(text, str):
                lines.append(text)
            continue
        text = getattr(obj, "text", None)
        if isinstance(text, str):
            lines.append(text)
    return "\n\n".join(lines)


def _format_context_item(index: int, item: object) -> str:
    """Render one context item for `/debug` output."""
    if isinstance(item, Message):
        return f"[{index}] Message role={item.role}\n{item.content}"
    if isinstance(item, ToolCallInput):
        return f"[{index}] ToolCallInput name={item.name} call_id={item.call_id} id={item.id}\n" f"{item.arguments}"
    if isinstance(item, ToolCallOutput):
        return (
            f"[{index}] ToolCallOutput call_id={item.call_id} success={item.result.success}\n" f"{item.result.content}"
        )
    if isinstance(item, Reasoning):
        reasoning = _extract_reasoning_text(item)
        return f"[{index}] Reasoning id={item.id}\n{reasoning or '<empty>'}"
    return f"[{index}] {item!r}"


def format_debug_dump(session: Session) -> str:
    """Build a structured debug dump for the current session."""
    lines: list[str] = []
    lines.append("=== Session Debug ===")
    lines.append(f"model: {session.model}")
    lines.append(f"instructions_as_system: {session.instructions_as_system}")
    lines.append(f"yolo: {session.yolo}")
    lines.append(f"reasoning_effort: {session.reasoning_effort}")
    lines.append(f"cwd: {session.cwd}")
    lines.append(f"auto_compact_token_limit: {session.auto_compact_token_limit}")
    lines.append(f"last_token_usage_total_tokens: {session.last_token_usage_total_tokens}")
    lines.append(f"compaction_count: {session.compaction_count}")
    lines.append(f"context_items: {len(session.context)}")
    lines.append("")
    lines.append("--- instructions ---")
    lines.append(session.instructions or "<empty>")
    lines.append("")
    lines.append("--- compaction prompt ---")
    lines.append(SUMMARIZATION_PROMPT)
    lines.append("")
    lines.append("--- context ---")
    if not session.context:
        lines.append("<empty>")
    else:
        for index, item in enumerate(session.context):
            lines.append(_format_context_item(index, item))
            lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    """CLI entry point for interactive runs."""
    project_root = Path(__file__).resolve().parents[2]
    default_system_prompt_path = project_root / "configs" / "tool_use_instructions.txt"
    parser = argparse.ArgumentParser(prog="nanocli")
    parser.add_argument("--model", required=True, help="Model name.")
    # Prompt cache retention is intentionally omitted to allow API defaults.
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Run bash commands without confirmation (dangerous).",
    )
    parser.add_argument(
        "--instructions-path",
        type=Path,
        default=default_system_prompt_path,
        help="Path to system prompt text file.",
    )
    # Having this arg because some API endpoints such as OpenRouter's gpt-oss wouldn't behave correctly
    # with the usual `instructions` parameter
    parser.add_argument(
        "--instructions-as-system",
        action="store_true",
        help="Send instructions as a system message in the input list.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="none",
        help="Reasoning effort (if applicable)",
    )
    parser.add_argument(
        "--auto-compact-token-limit",
        type=int,
        default=None,
        help="Token threshold that triggers automatic context compaction.",
    )
    args = parser.parse_args()

    if args.yolo:
        print("WARNING: --yolo enabled. Shell commands will run without confirmation.", file=sys.stderr)

    instructions = args.instructions_path.read_text(encoding="utf-8")
    session = Session(
        model=args.model,
        instructions=instructions,
        instructions_as_system=args.instructions_as_system,
        yolo=args.yolo,
        reasoning_effort=args.reasoning_effort,
        auto_compact_token_limit=args.auto_compact_token_limit,
    )

    print("Nano agent ready.")
    while True:
        try:
            line = input(format_prompt("User")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/debug":
            print_labeled("Debug", format_debug_dump(session))
            continue
        run_and_print(session, line)


if __name__ == "__main__":
    main()
