# HW4: Context Compaction

[CS598-LMZ Spring 2026](https://github.com/lingming/software-agents)

## Environment setup

### Install `uv`

We use [`uv`](https://github.com/astral-sh/uv) to manage Python environments and dependencies. `uv` is a super fast package / project manager written in Rust, similar to `cargo` for Rust. It replaces tools like `pip` and `conda` and is much faster.

Always use `uv run` to run commands inside the environment. For example, you can run `uv run python` to start a Python REPL inside the environment, or `uv run ipython` for a better interface. The environment will be automatically solved and activated when necessary.

### Install `prek`

We use [`prek`](https://github.com/j178/prek) as an alternative to [`pre-commit`](https://github.com/pre-commit/pre-commit) as it is much faster. Install `prek` simply with `uv tool install prek` (recommended) or follow its README.

### Run unit tests

```bash
uv run pytest
```

Initially, tests may fail at the `# STUDENT TODO` locations in `src/nanocli/core.py`. To connect your implementation to the tests, complete the TODO-marked functions.

### Run `prek`

```bash
# note: prek is installed as a global tool, so no need to use `uv run` here
prek run --all-files
```
