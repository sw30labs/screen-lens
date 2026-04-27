# Repository Guidelines

## Project Structure & Module Organization

ScreenLens is a Python 3.11+ package for local video scene intelligence. Core code lives in `src/`: `cli.py` exposes the Typer CLI, `pipeline.py` wires LangGraph flows, and modules such as `frame_extractor.py`, `captioner.py`, `embedder.py`, `vector_store.py`, `reconstruct.py`, and `config.py` own individual pipeline stages. Tests live in `tests/`, with YAML scenarios in `tests/test_cases.yaml`. Utility and benchmarking scripts live in `scripts/`. Static README assets live in `assets/`. Generated outputs, model artifacts, videos, and local databases belong in ignored paths such as `data/`, `OUTPUT/`, `ratita/`, and `input-videos/`.

## Build, Test, and Development Commands

Install the package locally:

```bash
pip install -e .
```

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest tests/test_pipeline.py -v
```

Run the CLI directly during development:

```bash
python -m src.cli info
python -m src.cli ingest "video.mov"
python -m src.cli search "What application is shown?"
```

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, clear type hints where useful, and concise docstrings for public classes, functions, and non-obvious helpers. Prefer `Path` for filesystem work and Pydantic config models from `src/config.py` instead of scattered constants. Keep module names lowercase with underscores, test functions named `test_*`, and classes named with `CamelCase`. No formatter or linter is currently configured in `pyproject.toml`; keep imports organized and run tests before submitting.

## Testing Guidelines

Tests use `pytest`, `pytest-asyncio`, and `pyyaml` from the `dev` extra. Add focused tests beside related coverage in `tests/test_pipeline.py` or split into new `tests/test_<module>.py` files as coverage grows. Prefer temporary directories and mocks for ChromaDB, CLIP, oMLX, Ollama, and video processing so tests stay local and repeatable. Avoid committing generated frames, captions, embeddings, videos, or model caches.

## Commit & Pull Request Guidelines

Recent history mostly follows Conventional Commit style, for example `feat(assemble): ...`, `fix(reconstruct): ...`, and `refactor(reconstruct): ...`. Use short, imperative commit subjects with a scope when helpful. Pull requests should include a summary, test results, linked issues when applicable, and screenshots or terminal output for CLI/user-visible behavior changes. Note any large model, hardware, or Apple Silicon assumptions.

## Agent-Specific Instructions

Keep edits narrowly scoped and preserve local artifacts. Do not move generated data into version control. When changing pipeline behavior, update README examples or this guide if developer workflows change.
