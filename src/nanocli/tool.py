"""Tool protocol, registry and local tool implementations for nanocli.

This module provides:

- A tiny `Tool` protocol that exposes:
  - `spec`: the Responses API tool schema (`FunctionToolParam`)
  - `call(...)`: the local implementation for a single tool call (JSON args in, ToolResult out)
  - `is_read_only()`: whether this tool should bypass approval prompting
- A global registry for dispatching tool calls by name.
- Tool implementations (`bash`, `read_file`, `list_dir`, `edit_file`).
"""

import functools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai.types.responses import FunctionToolParam


@dataclass(frozen=True)
class ToolResult:
    """A tool execution result suitable for client-side display.

    - `success=True` means the tool ran and produced output in `content`.
    - `success=False` means the tool failed; `content` is a human-readable error message.
    """

    success: bool
    content: str


class Tool[TArgs](Protocol):
    @property
    def spec(self) -> FunctionToolParam:
        """The tool schema to register with the Responses API."""
        ...

    def parse(self, arguments_json: str) -> TArgs | str:
        """
        Parse the JSON arguments into a structured object.
        Return a string error message on failure.
        """
        ...

    def call_typed(self, args: TArgs, cwd: Path) -> ToolResult:
        """
        The typed tool implementation. `args` is the structured output of `parse()`.
        `cwd` is the current working directory for this session.
        """
        ...

    def is_read_only(self) -> bool:
        """Whether this tool is read-only (i.e. guaranteed to have no side effects)."""
        return False

    def call(self, arguments_json: str, cwd: Path) -> ToolResult:
        """
        Default untyped call implementation that parses arguments and calls the
        typed implementation, returning parsing errors as tool errors.
        """
        args_or_error = self.parse(arguments_json)
        if isinstance(args_or_error, str):
            return ToolResult(success=False, content=args_or_error)
        return self.call_typed(args_or_error, cwd)


TOOL_REGISTRY: dict[str, Tool[Any]] = {}


def register_tool(tool: Tool[Any]) -> None:
    TOOL_REGISTRY[tool.spec["name"]] = tool
    # Keep tool_specs() correct even if tools are registered after the first call.
    tool_specs.cache_clear()


@functools.cache
def tool_specs() -> list[FunctionToolParam]:
    return [tool.spec for tool in TOOL_REGISTRY.values()]


def get_tool(name: str) -> Tool[Any] | None:
    return TOOL_REGISTRY.get(name)


def dispatch_tool_call(*, name: str, arguments_json: str, cwd: Path) -> ToolResult:
    tool = get_tool(name)
    if tool is None:
        return ToolResult(success=False, content=f"unknown tool: {name}")
    return tool.call(arguments_json, cwd=cwd)


def resolve_path(cwd: Path, user_path: str) -> Path:
    """Resolve `user_path` against `cwd`.

    - Relative paths are resolved as `cwd / user_path`.
    - Absolute paths are used directly.
    """
    p = Path(user_path)
    if p.is_absolute():
        return p
    return cwd / p


@dataclass(frozen=True)
class BashArgs:
    command: str


class BashTool(Tool[BashArgs]):
    @property
    def spec(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": "bash",
            "description": "Run a shell command in the project working directory.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to run (executed with shell=True). "
                            "Runs in Session.cwd. Output is returned as a text block containing "
                            "<stdout>, <stderr>, and <returncode> tags."
                        ),
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        }

    def parse(self, arguments_json: str) -> BashArgs | str:
        args = json.loads(arguments_json or "{}")
        command = args.get("command")
        if not isinstance(command, str):
            return "invalid arguments"
        return BashArgs(command=command)

    def call_typed(self, args: BashArgs, cwd: Path) -> ToolResult:
        if not cwd.is_dir():
            return ToolResult(success=False, content=f"working directory not found: {cwd}")
        result = subprocess.run(
            args.command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
        )
        text = (
            """
<stdout>
{stdout}
</stdout>

<stderr>
{stderr}
</stderr>

<returncode>
{returncode}
</returncode>
""".strip()
        ).format(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        return ToolResult(success=True, content=text)

    def is_read_only(self) -> bool:
        return False


@dataclass(frozen=True)
class ReadFileArgs:
    path: str
    offset: int
    limit: int


class ReadFileTool(Tool[ReadFileArgs]):
    @property
    def spec(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": "read_file",
            "description": "Read a UTF-8 text file from disk, returning numbered lines (like `cat -n`).",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path to read. If relative, it is resolved against Session.cwd. "
                            "Absolute paths are allowed. The path must exist and be a file."
                        ),
                    },
                    # With `strict: true`, OpenAI requires every schema property key to be listed
                    # in `required`. Use `null` to express "use the default".
                    "offset": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": ("1-indexed starting line number. Use null to default to 1 (the first line)."),
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "description": ("Maximum number of lines to return. Use null to default to 2000 lines."),
                    },
                },
                "required": ["path", "offset", "limit"],
                "additionalProperties": False,
            },
        }

    def is_read_only(self) -> bool:
        return True

    def parse(self, arguments_json: str) -> ReadFileArgs | str:
        # raise NotImplementedError
        args = json.loads(arguments_json or "{}")
        path = args.get("path")
        offset = args.get("offset")
        limit = args.get("limit")

        if not isinstance(path, str):
            return "invalid arguments"
        if offset is None or offset < 1:
            offset = 1
        elif not isinstance(offset, int):
            return "invalid arguments"
        if limit is None or limit < 0:
            limit = 2000
        elif not isinstance(limit, int):
            return "invalid arguments"
        return ReadFileArgs(path=path, offset=offset, limit=limit)

    def call_typed(self, args: ReadFileArgs, cwd: Path) -> ToolResult:
        # raise NotImplementedError
        resolved_path = resolve_path(cwd, args.path)
        if not resolved_path.exists() or not resolved_path.is_file():
            return ToolResult(success=False, content="not found")

        lines = []
        with open(resolved_path, encoding="utf-8") as f:
            all_lines = f.readlines()
        start_line = args.offset - 1
        end_line = min(start_line + args.limit, len(all_lines))

        for i, line in enumerate(all_lines[start_line:end_line], args.offset):
            lines.append(f"{i:6}\t{line}")
        return ToolResult(success=True, content="".join(lines))


@dataclass(frozen=True)
class ListDirArgs:
    path: str


class ListDirTool(Tool[ListDirArgs]):
    @property
    def spec(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": "list_dir",
            "description": "List a directory.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path to list. If relative, it is resolved against Session.cwd. "
                            "Absolute paths are allowed. The path must exist and be a directory."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        }

    def is_read_only(self) -> bool:
        return True

    def parse(self, arguments_json: str) -> ListDirArgs | str:
        # raise NotImplementedError
        args = json.loads(arguments_json or "{}")
        path = args.get("path")
        if not isinstance(path, str):
            return "invalid arguments"
        return ListDirArgs(path=path)

    def call_typed(self, args: ListDirArgs, cwd: Path) -> ToolResult:
        # raise NotImplementedError
        resolved_path = resolve_path(cwd, args.path)
        if not resolved_path.exists() or not resolved_path.is_dir():
            return ToolResult(success=False, content="tool error")
        entries = sorted(resolved_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        return ToolResult(success=True, content="\n".join(e.name + ("/" if e.is_dir() else "") for e in entries))


@dataclass(frozen=True)
class EditFileArgs:
    path: str
    search: str
    replace: str


class EditFileTool(Tool[EditFileArgs]):
    @property
    def spec(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": "edit_file",
            "description": "Edit a UTF-8 text file using search/replace (search must match exactly once).",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path to edit. If relative, it is resolved against Session.cwd. "
                            "Absolute paths are allowed. The path must exist and be a file."
                        ),
                    },
                    "search": {
                        "type": "string",
                        "description": (
                            "Substring to find. Must be non-empty. The substring must match exactly once in the file; "
                            "returns an error if it matches 0 times or multiple times."
                        ),
                    },
                    "replace": {"type": "string", "description": "Replacement string."},
                },
                "required": ["path", "search", "replace"],
                "additionalProperties": False,
            },
        }

    def is_read_only(self) -> bool:
        return False

    def parse(self, arguments_json: str) -> EditFileArgs | str:
        # raise NotImplementedError
        args = json.loads(arguments_json or "{}")
        path = args.get("path")
        search = args.get("search")
        replace = args.get("replace")
        if not all(isinstance(x, str) for x in [path, search, replace]):
            return "invalid arguments"
        return EditFileArgs(path=path, search=search, replace=replace)

    def call_typed(self, args: EditFileArgs, cwd: Path) -> ToolResult:
        # raise NotImplementedError
        resolved_path = resolve_path(cwd, args.path)
        if not resolved_path.exists() or not resolved_path.is_file():
            return ToolResult(success=False, content="tool error")
        if args.search == "":
            return ToolResult(success=False, content="non-empty")

        content = resolved_path.read_text(encoding="utf-8")
        occurrences = content.count(args.search)
        if occurrences == 0:
            return ToolResult(success=False, content="not found")
        if occurrences > 1:
            return ToolResult(success=False, content=f"exactly once match required, but found {occurrences} matches")

        new_content = content.replace(args.search, args.replace)
        resolved_path.write_text(new_content, encoding="utf-8")
        return ToolResult(success=True, content=f"{occurrences} replacements")


register_tool(BashTool())
register_tool(ReadFileTool())
register_tool(ListDirTool())
register_tool(EditFileTool())
