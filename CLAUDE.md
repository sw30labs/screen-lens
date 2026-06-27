# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScreenLens is a local video scene intelligence pipeline for Apple Silicon. It ingests screen recordings, extracts keyframes, generates dense captions with a vision-language model, embeds them with CLIP, stores them in ChromaDB, and answers natural-language queries — all locally. A second pipeline (`reconstruct`) uses the captions to rebuild the original artifacts (Python files, Markdown docs, PDFs, GUI walkthroughs) shown in a recording. A third pipeline (`transcribe`) is the verbatim path: it OCRs frames character-for-character with a vision model and stitches them in text space to faithfully reproduce the text/code shown in a scrolling recording.

## Common Commands

```bash
# Install (editable)
pip install -e .

# Ingest a single video (default: keyframe extraction + Qwen3.5-VL via oMLX)
python -m src.cli ingest "video.mov"

# Ingest with Ollama backend instead of oMLX
python -m src.cli ingest "video.mov" --backend ollama --strategy fixed_fps --fps 1.0

# Use a specific oMLX model
python -m src.cli ingest "video.mov" --omlx-model mlx-community/Qwen3.5-35B-A3B-4bit

# Batch-ingest a folder (each video gets its own ./data/<slug>/ collection)
python -m src.cli batch "/path/to/recordings/"

# Search the most recently configured collection
python -m src.cli search "What application is being demonstrated?"

# Ingest + search in one shot
python -m src.cli run "video.mov" "Summarize what happens"

# Full-video summary from all captions (single-pass or hierarchical chunking)
python -m src.cli summarize

# Reconstruct artifacts from all ingested folders under ./data/
python -m src.cli reconstruct

# Verbatim transcription (frame_select → vision OCR → text-space stitch)
# LLM seam/indent cleanup is OFF by default; opt in with --cleanup.
python -m src.cli transcribe "video.mov"
python -m src.cli transcribe "code.mov" --deterministic   # Apple Vision cross-check (needs: pip install ocrmac)
python -m src.cli transcribe "doc.mov" --cleanup          # run the optional LLM seam/indent pass

# List served oMLX models, labeled vision / text-only / draft
python -m src.cli models

# Vector store stats
python -m src.cli info

# Tests
pytest tests/ -v
pytest tests/test_pipeline.py::TestEmbedder -v        # one class
pytest tests/test_pipeline.py::TestEmbedder::test_embed_text -v   # one test
```

`ffmpeg` must be on PATH (`brew install ffmpeg`). oMLX must be running for the default backend, with `MLX_API_KEY`/`OMLX_API_KEY` set if authentication is enabled. The Ollama caption fallback requires `ollama pull llama3.2-vision`.

## Architecture

The codebase is two LangGraph `StateGraph` pipelines that share a single `ScreenLensConfig` (Pydantic) and the same oMLX client adapter.

### Pipeline 1 — Ingest / Search (`src/pipeline.py`)

`StateGraph` over a `ScreenLensState` TypedDict. Three graph builders:

- `build_ingest_graph()`  — `ingest → caption → embed`
- `build_search_graph()`  — `search → summarize`
- `build_full_graph()`    — both chained end-to-end

Per-stage modules:

| Node | Module | Purpose |
|---|---|---|
| `ingest_node` | `frame_extractor.py` | Hybrid keyframe detection (SSIM + pHash + HSV histogram, OpenCV) or fixed-FPS fallback. Decision logic in `_extract_keyframes` — emits when any signal trips AND `min_interval_seconds` has elapsed, or unconditionally every `max_interval_seconds`. |
| `caption_node` | `captioner.py` | Backend factory (`_get_captioner`). `OMLXCaptioner` sends OpenAI-compatible vision requests to oMLX. `OllamaCaptioner` uses `langchain_ollama.ChatOllama` with base64-encoded images. |
| `embed_node` | `embedder.py`, `vector_store.py` | OpenCLIP `ViT-B-32` on `mps`, then writes embeddings + metadata into ChromaDB. |
| `search_node` | `vector_store.py` | Encodes query text via the same CLIP, ChromaDB cosine search. |
| `summarize_node` | `langchain_ollama.ChatOllama` | Summarizes top-k results into a natural-language answer. |
| `summarize_all_node` | `pipeline.py` | **Different code path** — full-video summary using oMLX. Uses `captioning.omlx_model_context` and chooses single-pass vs. hierarchical chunking via `_compute_chunk_strategy`. |

State flows by returning partial dicts that LangGraph merges; `elapsed_seconds` accumulates per-stage timings.

### Pipeline 2 — Reconstruct (`src/reconstruct.py`)

A more sophisticated graph: `classify → plan → (parallel workers | sequential) → qa_reflect → save`, with a retry edge from `qa_reflect → plan` that can loop up to `MAX_QA_ITERATIONS = 3`.

Key mechanics worth knowing before editing:

- **Parallel fan-out via `Send`.** `route_to_workers` returns either a list of `Send("reconstruct_worker", ...)` payloads (parallel) or the literal string `"reconstruct_sequential"` (single node). Parallel is only chosen when `parallel_safe=True` AND there is more than one task. The planner sets `parallel_safe` per content type — Python files only when the LLM judges them independent (no cross-imports), GUI demos always (walkthrough + reference are independent), Markdown/PDF never.
- **Reducer for collecting sub-agent outputs.** `artifacts: Annotated[list[dict], operator.add]` lets multiple `Send`-dispatched workers append their results without clobbering each other. Each artifact carries an `iteration` field so `qa_reflect_node` and `save_node` can distinguish current-iteration outputs from prior-iteration ones.
- **Client cache.** `_MODEL_CACHE` (module-level dict, keyed by oMLX URL and model id) ensures the oMLX client is reused across reconstruction nodes.
- **Reflection feedback loop.** When `qa_reflect_node` fails, it stores `qa_feedback` and increments `qa_iteration`; the next pass through `plan_node` injects `PREVIOUS QA FEEDBACK` into each task prompt. After `MAX_QA_ITERATIONS - 1`, QA force-passes to avoid infinite loops.
- **JSON parsing.** LLM JSON responses are parsed via `parse_json_response`, which tries direct → fenced → first `{...}` block. Always use this helper rather than `json.loads` directly on model output.

### Pipeline 3 — Transcribe (`src/transcribe.py`)

The verbatim path. Not a `StateGraph` — a straight function pipeline:
`select_frames → VerbatimOCR → stitch_frames → (optional) LLM cleanup → output/transcript.md`.

| Stage | Module | Purpose |
|---|---|---|
| frame select | `frame_select.py` | Dense sample (default 2 fps), drop ONLY near-exact static duplicates (SSIM > 0.992). On scrolling text, pixel metrics can't tell "new content" from "same content shifted", so the real dedup happens later in text space. |
| verbatim OCR | `ocr.py` | `VerbatimOCR` sends each frame to a **vision** model via oMLX and copies text character-for-character (never describes). Hard capability guard + live probe abort if the model is text-only. Anti-loop sampler controls; optional Apple Vision (`ocrmac`) deterministic backstop for code. |
| stitch | `stitch.py` | Text-space dedup of scroll overlap (fuzzy line canonicalization + difflib matching-blocks), strips page headers/footers. Writes `transcript.raw.md`. |
| cleanup (optional) | `transcribe.py` | LLM seam/indent repair only. **Off by default.** |

Key mechanics worth knowing before editing:

- **Thinking is disabled for OCR and cleanup.** Both pass `chat_template_kwargs={"enable_thinking": false}` (gated by `OCRConfig.disable_thinking` / `ReconstructionConfig.disable_thinking`, default true). A reasoning model used for verbatim OCR otherwise spends the entire `max_tokens` budget on chain-of-thought and never emits the transcription. `omlx_client.strip_thinking` also strips complete/dangling `<think>` blocks, and `_post_chat` warns on `finish_reason == "length"`.
- **Cleanup is OFF by default** (`ReconstructionConfig.enabled = False`; CLI opt-in via `--cleanup`). The raw stitched OCR is already verbatim; an LLM tends to drop content while "repairing".
- **Cleanup coverage guard.** When cleanup runs, chunk input is bounded by the output token cap (not just the context window) so a chunk can't truncate mid-output. After each chunk, `_chunk_coverage` checks the fraction of distinct input lines that survived; below `MIN_CHUNK_COVERAGE` (0.97) the LLM output is discarded and the **raw stitched chunk is kept** — `transcript.md` can never lose content vs. `transcript.raw.md`.
- **Two models, two jobs.** OCR uses a vision model (`OCRConfig.model` → `OCR_MODEL` env → `RECOMMENDED_OCR_MODEL = "Qwen3.6-27B-bf16"`); cleanup uses a text model (`ReconstructionConfig.model` → `LLM_MODEL`/`MLX_MODEL` env). They have separate `OMLXClient`s built via `from_endpoint`.

### Configuration (`src/config.py`)

A single `ScreenLensConfig` Pydantic model composed of `FrameExtractionConfig`, `CaptioningConfig`, `EmbeddingConfig`, `VectorDBConfig`, `SearchConfig`, plus the transcribe-path models `OCRConfig`, `FrameSelectionConfig`, `ReconstructionConfig`. Two enums: `ExtractionStrategy` (`keyframe` | `fixed_fps`) and `CaptionBackend` (`omlx` | `ollama`). The CLI mutates this config in place from command-line flags before passing `config.model_dump()` into the graph state.

Set `MLX_MODEL`, `OMLX_MODEL`, or `--omlx-model` to choose an oMLX-served model. For the transcribe path, set `OCR_MODEL` (vision) separately from `LLM_MODEL` (text cleanup).

### Data Layout

Each ingested video gets its own folder under `./data/<video_slug>/` (set in `cli.py::batch`):

```
data/<slug>/
  frames/                 # extracted keyframe / sampled JPGs (PNGs for transcribe)
  captions/
    caption_NNNNNN.json   # one per frame (ingest path)
    all_captions.json     # combined — what reconstruct/summarize read
  ocr/
    ocr_NNNNNN.json       # one per frame (transcribe path)
    all_ocr.json          # combined verbatim OCR
  chromadb/               # per-video persistent collection
  output/                 # reconstruct.py and transcribe.py write here
    reconstruction_meta.json
    transcript.raw.md     # stitched verbatim OCR (faithful, no LLM)
    transcript.md         # == raw unless --cleanup ran
    transcribe_meta.json
```

`transcribe` always derives a fresh timestamped slug (`<stem>_<YYYYMMDD_HHMMSS>`), so re-running never reuses a prior folder.

The single-video commands (`ingest`, `search`) default to the top-level `./data/` instead of a slugged subfolder, so running `ingest` followed by `batch` will mix collections — `batch` is the canonical path for multi-video work.

## Things to Watch For

- **CLIP device** defaults to `mps`. Tests pin it to `cpu` (`TestEmbedder` fixture) so they run anywhere. If you add embedder tests, do the same.
- **`data/` is gitignored** along with `*.mov`/`*.mp4` and other video formats — don't try to commit fixtures.
- **oMLX context planning** uses `captioning.omlx_model_context` to chunk long caption streams. Keep it conservative unless the served model's context is known.
- **Transcribe OCR must use a vision model.** `RECOMMENDED_OCR_MODEL = "Qwen3.6-27B-bf16"`. A text-only `OCR_MODEL` aborts via the capability guard + live probe.
- **`disable_thinking` stays on for OCR/cleanup.** Turning it off with a reasoning model regresses to truncated, all-reasoning output (see `tests/test_transcribe.py::test_strip_thinking_handles_truncated_open_tag`).
- **Don't size cleanup chunks past the output token cap.** Chunk input is intentionally bounded by `max_tokens`, not just `model_context` — otherwise output truncates mid-chunk and silently loses content.
- **oMLX serializes requests.** Probing the server while a long job runs queues behind it (a 3-word call can take minutes). The model fits the memory ceiling matters: `MiniMax-M3-4bit` is ~236 GB and won't load under a 71 GB ceiling; `Qwen3.6-27B-bf16` (~54 GB) fits.

## Coding Guidelines
Behavioral guidelines to reduce common LLM coding mistakes, on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
