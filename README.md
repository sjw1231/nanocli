# nanocli

A minimal command-line code agent inspired by [Codex](https://github.com/openai/codex) and [Claude Code](https://github.com/anthropics/claude-code). Built for [CS598-LMZ Spring 2026](https://github.com/lingming/software-agents).

## Features

- **ReAct agentic loop**: the model proposes actions, the harness executes them locally, and observations are fed back until no new tool calls remain
- **Structured tool calling**: uses the OpenAI Responses API `function_call` / `function_call_output` interface with four built-in tools: `bash`, `read_file`, `list_dir`, and `edit_file`
- **Agent skills**: loads skill definitions from `skills/<skill-name>/SKILL.md`; skills can be activated implicitly (metadata always in context) or explicitly (user prefixes message with `$skill-name` to inject full instructions)
- **Automatic context compaction**: tracks `usage.total_tokens` from each API response; when the configured token limit is reached, asks the model for a handoff summary and replaces old history with a compact checkpoint before continuing
- **Approval gate**: non-read-only tool calls prompt for confirmation unless `--yolo` is set

## Environment Setup

### Install `uv`

[`uv`](https://github.com/astral-sh/uv) manages the Python environment and dependencies. Always use `uv run` to execute commands inside the project.

### Install `prek`

```bash
uv tool install prek
```

[`prek`](https://github.com/j178/prek) handles linting and formatting checks.

### Run unit tests

```bash
uv run pytest
```

### Run linter / formatter

```bash
prek run --all-files
```

## Usage

Start the agent (requires `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`):

```bash
uv run nanocli
```

Key flags:

| Flag | Description |
|---|---|
| `--model <name>` | Model to use (e.g. `gpt-4o`, `mock`) |
| `--yolo` | Skip approval prompts for all tool calls |
| `--instructions-as-system` | Send instructions as a system message instead of the `instructions` field |
| `--auto-compact-token-limit <n>` | Token threshold that triggers automatic context compaction |

## Project Structure

```
src/nanocli/
  cli.py            # Thin CLI: renders events, handles approval prompts
  core.py           # Agent loop: run_turn(), context compaction
  tool.py           # Tool registry: bash, read_file, list_dir, edit_file
  skill.py          # Skill discovery, rendering, and explicit resolution
  _dev/
    mock_server.py  # Local Responses API mock for offline testing
skills/             # Skill definitions (SKILL.md files)
tests/              # Unit tests
```

## Architecture

```
User input
   |
   v
CLI (cli.py)
   |
   v
maybe_compact_session()   <-- checks token usage, rewrites history if limit reached
   |
   v
core.run_turn(...)  -- yields -->  ChatItem / ToolApprovalRequest
   |                                        |
   |                                        v
   |                              TUI renders / asks approval
   |
   v
OpenAI Responses API  (tools + skill context in request)
   |
   v
dispatch_tool_call() -> ToolCallOutput -> appended to context
   |
   v
repeat until no new tool calls
```
