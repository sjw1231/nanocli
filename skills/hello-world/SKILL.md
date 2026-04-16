---
name: hello-world
description: Use a tiny, deterministic workflow to create or run a minimal "hello world" example in the language the user asks for.
---

# Hello World

Use this skill when the user asks for a first-run example, starter snippet, or quick sanity check.

## Goal

Produce the smallest runnable "hello world" result for the requested language or environment.

## Workflow

1. Detect the target language/runtime from user text.
2. Prefer a direct command-line one-liner first (no file creation).
3. Create a file only if one-liner execution is not practical for that language/runtime or the user explicitly asks for a file.
4. Run the example if a runtime is available.
5. Report exact command(s) used and output.

## Defaults

- Python: `python -c 'print("hello world")'`
- JavaScript (Node): `node -e 'console.log("hello world")'`
- Bash: `echo "hello world"`

If the user does not specify a language, default to Python.

## Constraints

- Keep the example intentionally small.
- Prefer no-file execution whenever possible.
- Avoid adding frameworks, package managers, or extra project setup unless explicitly requested.
- If execution is unavailable, provide the exact command the user should run (and only include file steps if they are truly needed).
