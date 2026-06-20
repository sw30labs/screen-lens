# ScreenLens redesign — verbatim transcription path

## TL;DR of why it was failing

The pipeline was sound on paper but **the OCR step was reading every frame
blind.** `.env` had `MLX_MODEL=MiniMax-M2.7`, a **text-only** model, and the
captioner sent it images. MiniMax can't see pixels, so all 173 captions came
back *"No image or video frame has been provided"* — and every downstream stage
(embedding, reconstruction) faithfully processed that emptiness. The hybrid
SSIM/pHash keyframe logic and the difflib dedup were never the bottleneck.

A second, independent confirmation came from measuring pixel metrics on the real
frames: on scrolling dense text, **SSIM/pHash cannot tell "new content" from
"same content shifted up a few rows"** — overlap-SSIM stays ~0.5–0.7 even for a
clean scroll. That is why deterministic frame dedup "failed": it was the wrong
tool for the job. Pixels reliably detect exactly one thing here — *near-exact
static duplicates*.

## The new approach (verbatim, scroll-aware)

```
video.mov
  → frame_select.py   sample densely (2 fps), drop ONLY near-exact dupes
  → ocr.py            verbatim OCR via a VISION model (transcribe, don't describe)
  → stitch.py         text-space dedup: fuzzy-canonicalize lines, difflib
                      matching-blocks find the scroll overlap, splice the new tail,
                      strip page headers/footers, majority-vote OCR flicker
  → transcribe.py     optional text-LLM cleanup, STRICTLY seams + indentation
  → data/<slug>/output/transcript.md
```

Two models, separated by job:

| Job | Model | Why |
|-----|-------|-----|
| **Read frames (OCR)** | a **vision** model — default `mlx-community/olmOCR-2-7B-1025-8bit` | purpose-built for dense document fidelity; never a text-only model again |
| **Clean seams (LLM)** | a **text** model — `MiniMax-M2` is fine here | it never sees images; tidying stitched text is what it's good at |

The dedup that matters happens **in text space, after OCR** — proven robust to
OCR flicker, dropped lines, and static-pause duplicate frames (see
`tests/test_transcribe.py`).

## What changed in the code

New modules:
- `src/frame_select.py` — scroll-safe frame selection (dense sample + drop near-exact dupes).
- `src/ocr.py` — `VerbatimOCR`: vision OCR with a hard capability guard + live probe, anti-loop sampler controls, and an optional Apple Vision (ocrmac) deterministic backstop for code.
- `src/stitch.py` — text-space stitcher (the core engine).
- `src/transcribe.py` — orchestrator + constrained LLM cleanup.

Changed:
- `src/config.py` — added `OCRConfig`, `FrameSelectionConfig`, `ReconstructionConfig`; verbatim OCR prompts.
- `src/omlx_client.py` — generic `from_endpoint` constructor, `list_models()`, real capability checks, sampler-param passthrough; added MiniMax/Kimi to the known-text-only guard list (VL variants still detected as vision).
- `src/cli.py` — new `transcribe` and `models` commands.
- `.env` / `.env.example` — split `OCR_MODEL` (vision) from `MLX_MODEL`/`LLM_MODEL` (text).

## Run it

```bash
# 1. See which of your oMLX models can actually OCR:
python -m src.cli models

# 2. Transcribe a recording (verbatim):
python -m src.cli transcribe input/policies.mov

# For code recordings, add the deterministic cross-check:
python -m src.cli transcribe input/code.mov --deterministic
#   (requires: pip install ocrmac)

# Output: data/<slug>/output/transcript.md  (+ transcript.raw.md, ocr/all_ocr.json)
```

The first thing `transcribe` does is **probe the OCR model with one real frame**;
if it answers as if no image was sent, it aborts immediately with an actionable
message instead of silently producing 173 empty files.

## Model selection — from YOUR oMLX inventory (June 2026)

You already have strong vision models served locally; no download needed. Their
names hide it (no "VL"), but Qwen3.6 and Gemma-4 are unified multimodal.

**OCR (vision) — pick one:**

| Model | Size | Why |
|-------|------|-----|
| `Qwen3.6-27B-bf16` | 51 GB | **Max fidelity.** Qwen-VL lineage is the OCR gold standard; bf16 = no quant loss. Default. |
| `gemma-4-31b-bf16` | 58 GB | Purpose-built for OCR / document / PDF / screen-UI parsing. Excellent alternative. |
| `Qwen3.6-35B-A3B-8bit-MTPLX-Optimized-Speed` | 37 GB | **Fastest.** MoE (~3B active) + speed-tuned — best for many frames. |

Text-only (do NOT use for OCR; fine for cleanup/analysis): MiniMax-M2/M3,
Kimi-K2.6/2.7, DeepSeek-V4, Nemotron-3, GLM-5.1. The `*-DFlash` / `*-MTP`
entries are speculative-decode draft models, not standalone.

**For code recordings:** keep `--deterministic` on. VLMs hallucinate code tokens
(linguistic-prior drift); Apple Vision never invents a line, so a character-level
disagreement is flagged for review rather than silently wrong.

**LLM cleanup (text):** any capable text model. `Kimi-K2.7-Code` is a great
choice for code recordings; `MiniMax-M2` is fine for prose. It only fixes seams
and indentation — prompted never to re-invent content.

> The default OCR model is just a starting point. `screenlens models` now labels
> each served model vision / text-only / draft, and `transcribe` probes the model
> with one real frame before processing — so a wrong choice fails instantly with
> a clear message instead of producing empty output.
