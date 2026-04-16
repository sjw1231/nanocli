"""Unit tests for nanocli tools and tool schemas.

These tests are intentionally small and explicit.
Treat them as the specification for tool behavior in HW2.
"""

from __future__ import annotations

import json
from pathlib import Path

from nanocli.tool import (
    ReadFileTool,
    dispatch_tool_call,
    resolve_path,
    tool_specs,
)


def test_tool_params_are_strict_and_disallow_additional_properties() -> None:
    """Tool schemas must be strict JSON Schemas (OpenAI `strict: true` tools)."""
    tools = tool_specs()
    names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
    assert {"bash", "read_file", "list_dir", "edit_file"} <= names
    for tool in tools:
        assert tool["type"] == "function"
        assert tool["strict"] is True
        raw_params = tool.get("parameters")
        assert isinstance(raw_params, dict)
        assert raw_params.get("type") == "object"
        assert raw_params.get("additionalProperties") is False
        # OpenAI strict tool schemas require every property key to be listed in `required`.
        props = raw_params.get("properties")
        required = raw_params.get("required")
        assert isinstance(props, dict)
        assert isinstance(required, list)
        assert set(required) == set(props.keys())


def test_resolve_path_handles_relative_and_absolute_paths(tmp_path: Path) -> None:
    """Paths can be relative to `cwd` or absolute paths."""
    assert resolve_path(tmp_path, "marker.txt") == tmp_path / "marker.txt"
    marker = tmp_path / "abs.txt"
    assert resolve_path(tmp_path, str(marker)) == marker


def test_parse_invalid_arguments_returns_error_string() -> None:
    """Tool parsing errors should be returned as strings (not raised)."""
    out = ReadFileTool().parse(json.dumps({"path": 123, "offset": None, "limit": None}))
    assert out == "invalid arguments"


def test_bash_returns_structured_text(tmp_path: Path) -> None:
    """bash returns a tagged text block with stdout/stderr/returncode."""
    output = dispatch_tool_call(name="bash", arguments_json=json.dumps({"command": "echo hi"}), cwd=tmp_path)
    assert output.success is True
    assert "<stdout>" in output.content
    assert "<stderr>" in output.content
    assert "<returncode>" in output.content


def test_bash_bad_cwd_returns_error(tmp_path: Path) -> None:
    """bash should return a tool error when its working directory does not exist."""
    bad_root = tmp_path / "missing"
    output = dispatch_tool_call(name="bash", arguments_json=json.dumps({"command": "echo hi"}), cwd=bad_root)
    assert output.success is False


def test_read_file_slices_by_offset_and_limit(tmp_path: Path) -> None:
    """read_file supports 1-indexed `offset` and a `limit` in number of lines."""
    marker = tmp_path / "marker.txt"
    marker.write_text("a\nb\nc\n", encoding="utf-8")
    output = dispatch_tool_call(
        name="read_file",
        arguments_json=json.dumps({"path": "marker.txt", "offset": 2, "limit": 1}),
        cwd=tmp_path,
    )
    assert output.success is True
    assert output.content == "     2\tb\n"


def test_read_file_defaults_offset_and_limit_when_null(tmp_path: Path) -> None:
    """read_file defaults offset=1 and limit=2000 when null, and prefixes line numbers like `cat -n`."""
    marker = tmp_path / "marker.txt"
    marker.write_text("a\nb\n", encoding="utf-8")
    output = dispatch_tool_call(
        name="read_file",
        arguments_json=json.dumps({"path": "marker.txt", "offset": None, "limit": None}),
        cwd=tmp_path,
    )
    assert output.success is True
    assert output.content == "     1\ta\n     2\tb\n"


def test_read_file_missing_returns_error(tmp_path: Path) -> None:
    """read_file returns a tool error when the path does not exist."""
    output = dispatch_tool_call(
        name="read_file",
        arguments_json=json.dumps({"path": "missing.txt", "offset": None, "limit": None}),
        cwd=tmp_path,
    )
    assert output.success is False
    assert "not found" in output.content


def test_list_dir_lists_sorted_entries(tmp_path: Path) -> None:
    """list_dir returns a newline-separated sorted listing (directories end with '/')."""
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a_dir").mkdir()
    output = dispatch_tool_call(name="list_dir", arguments_json=json.dumps({"path": "."}), cwd=tmp_path)
    assert output.success is True
    assert output.content.splitlines() == ["a_dir/", "b.txt"]


def test_edit_file_search_replace_single_match(tmp_path: Path) -> None:
    """edit_file replaces a search string that matches exactly once."""
    path = tmp_path / "t.txt"
    path.write_text("abc\n", encoding="utf-8")
    result = dispatch_tool_call(
        name="edit_file",
        arguments_json=json.dumps({"path": "t.txt", "search": "b", "replace": "X"}),
        cwd=tmp_path,
    )
    assert result.success is True
    assert "1 replacements" in result.content
    assert path.read_text(encoding="utf-8") == "aXc\n"


def test_edit_file_multiple_occurrences_returns_error(tmp_path: Path) -> None:
    """edit_file returns an error if the search string matches multiple times (ambiguous edit)."""
    path = tmp_path / "t.txt"
    path.write_text("a a a", encoding="utf-8")
    result = dispatch_tool_call(
        name="edit_file",
        arguments_json=json.dumps({"path": "t.txt", "search": "a", "replace": "b"}),
        cwd=tmp_path,
    )
    assert result.success is False
    assert "exactly once" in result.content
    assert "3" in result.content


def test_edit_file_empty_search_returns_error(tmp_path: Path) -> None:
    """edit_file rejects empty search strings."""
    path = tmp_path / "t.txt"
    path.write_text("hello", encoding="utf-8")
    result = dispatch_tool_call(
        name="edit_file",
        arguments_json=json.dumps({"path": "t.txt", "search": "", "replace": "x"}),
        cwd=tmp_path,
    )
    assert result.success is False
    assert "non-empty" in result.content


def test_edit_file_search_not_found_returns_error(tmp_path: Path) -> None:
    """edit_file returns a tool error when the search string is not found."""
    path = tmp_path / "t.txt"
    path.write_text("hello", encoding="utf-8")
    result = dispatch_tool_call(
        name="edit_file",
        arguments_json=json.dumps({"path": "t.txt", "search": "missing", "replace": "x"}),
        cwd=tmp_path,
    )
    assert result.success is False
    assert "not found" in result.content
