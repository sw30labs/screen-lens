"""
Frame Captioning Module — dual backend.

Backends:
  1. **mlx-vlm** (default) — Runs Qwen3.5-VL natively on Apple Silicon via MLX.
     Best quality + speed on M-series Macs. Auto-downloads model on first use.
  2. **ollama** (fallback) — Any Ollama vision model (llama3.2-vision, etc.).
     Works on any platform with Ollama installed.
"""
import base64
import json
import logging
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import CaptioningConfig, CaptionBackend

logger = logging.getLogger("screenlens.captioner")


# ── mlx-vlm enable_thinking=False patch ────────────────────────────────────
# mlx_vlm 0.4.4's _generate_batch calls apply_chat_template without forwarding
# kwargs, so enable_thinking=False can't be passed through batch_generate
# normally. Qwen3.5-VL's chat template prepends <think> to the assistant turn,
# so the model's output starts inside a think block and emits the closing
# </think> partway through — burning ~50% of max_tokens on planning prose,
# often truncating mid-thought before any structured response. This patch
# wraps the apply_chat_template binding inside the mlx_vlm.generate submodule
# to inject enable_thinking=False so the Qwen3.5 template skips thinking
# entirely.
#
# Kwarg flow: apply_chat_template → get_chat_template →
# processor.tokenizer.apply_chat_template(messages, ..., enable_thinking=False).
# HuggingFace tokenizers ignore unknown kwargs in their Jinja templates, so
# this is a no-op for non-Qwen models. The patch is idempotent and applied
# once at module import time.
def _patch_mlx_vlm_disable_thinking() -> None:
    try:
        import importlib
        gen_mod = importlib.import_module("mlx_vlm.generate")
    except ImportError:
        return  # mlx-vlm not installed; Ollama-only path still works
    if getattr(gen_mod, "_screenlens_thinking_patched", False):
        return
    _orig = gen_mod.apply_chat_template

    def _patched(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig(*args, **kwargs)

    gen_mod.apply_chat_template = _patched
    gen_mod._screenlens_thinking_patched = True
    logger.info("Patched mlx_vlm.generate.apply_chat_template with enable_thinking=False")


_patch_mlx_vlm_disable_thinking()


# ── MLX-VLM Backend ────────────────────────────────────────────────────────

class MLXVLMCaptioner:
    """Caption frames using Qwen3.5-VL (or any VLM) via mlx-vlm on Apple Silicon."""

    def __init__(self, config: CaptioningConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        from mlx_vlm import load

        # Determine what to pass to load():
        #   - If user gave an explicit local path, use that
        #   - Otherwise pass the repo_id directly — mlx_vlm handles download + cache
        model_id = self.config.mlx_model_path or self.config.mlx_repo_id

        logger.info(f"Loading MLX-VLM model: {model_id} ...")
        print(f"Loading MLX-VLM model: {model_id} (this may download on first use)")

        loaded = load(model_id, lazy=True)
        if isinstance(loaded, tuple):
            self._model, self._tokenizer = loaded[:2]
        else:
            raise RuntimeError(f"Unexpected return from mlx_vlm.load: {type(loaded)}")

        logger.info("MLX-VLM model loaded successfully.")

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3.5 thinking artifacts (defense-in-depth).

        Two cases handled:

        1. **Matched pair** ``<think>...</think>``. Legacy / non-batch path.

        2. **Orphan closing tag** — just ``</think>`` partway through. Qwen3.5's
           chat template prepends ``<think>`` to the assistant turn, so the
           model's output begins inside a think block and only emits the closing
           tag. Everything before that closing tag is thinking content.

        With ``_patch_mlx_vlm_disable_thinking`` active at module load, neither
        case should appear in fresh output, but this is the safety net for
        models or future mlx-vlm versions where the patch may silently no-op.
        """
        # Case 1: matched pairs
        cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        # Case 2: orphan closing tag — strip everything before and including it
        if '</think>' in cleaned:
            cleaned = cleaned.split('</think>', 1)[1]
        return cleaned.strip()

    def caption(self, image_path: str) -> str:
        """Generate a caption for a single frame."""
        from mlx_vlm import generate

        self._load_model()

        # Build chat prompt using the tokenizer's template
        # Qwen3.5 uses enable_thinking=False (NOT /no_think which was Qwen3 only)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.config.user_prompt},
                ],
            },
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3.5: disable chain-of-thought
        )

        result = generate(
            self._model,
            self._tokenizer,
            prompt,
            image=image_path,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # mlx_vlm.generate can return str, or object with .text
        if isinstance(result, str):
            raw = result
        elif hasattr(result, "text"):
            raw = result.text
        else:
            raw = str(result)

        # Safety net: strip any residual <think>...</think> blocks
        return self._strip_thinking(raw)

    def caption_batch(self, image_paths: list[str]) -> list[str]:
        """Generate captions for a batch of frames in a single forward pass.

        Uses ``mlx_vlm.batch_generate``, which packs multiple sequences into one
        batched forward pass with a shared KV cache. Same-shape images are
        grouped to avoid padding waste (``group_by_shape=True``).

        Note on chain-of-thought suppression: ``batch_generate`` does not
        forward kwargs to ``apply_chat_template``, so we cannot pass
        ``enable_thinking=False`` through. We instead inject the system prompt
        as a multi-turn message list and rely on ``_strip_thinking`` to scrub
        any residual ``<think>...</think>`` blocks (the system prompt also
        explicitly tells the model not to emit them).
        """
        from mlx_vlm import batch_generate

        self._load_model()

        if not image_paths:
            return []

        # Each prompt is a [system, user] message list. apply_chat_template
        # (called inside batch_generate) handles list-of-dicts and places the
        # image token on the user message via num_images=1.
        prompts = [
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": self.config.user_prompt},
            ]
            for _ in image_paths
        ]

        response = batch_generate(
            self._model,
            self._tokenizer,
            images=list(image_paths),
            prompts=prompts,
            max_tokens=self.config.max_tokens,
            verbose=False,
            group_by_shape=True,
        )

        return [self._strip_thinking(t or "") for t in response.texts]


# ── Ollama Backend ──────────────────────────────────────────────────────────

class OllamaCaptioner:
    """Caption frames using any Ollama vision model."""

    def __init__(self, config: CaptioningConfig):
        self.config = config

    def caption(self, image_path: str) -> str:
        """Generate a caption for a single frame."""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
        )

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            ("system", self.config.system_prompt),
            (
                "human",
                [
                    {"type": "text", "text": self.config.user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            ),
        ]

        response = llm.invoke(messages)
        return response.content

    def caption_batch(self, image_paths: list[str]) -> list[str]:
        """Sequential fallback: Ollama has no batch API, so we loop.

        Exists for call-site uniformity with MLXVLMCaptioner.caption_batch.
        """
        return [self.caption(p) for p in image_paths]


# ── Factory + Batch Processing ──────────────────────────────────────────────

def _get_captioner(config: CaptioningConfig):
    """Return the appropriate captioner backend."""
    if config.backend == CaptionBackend.mlx_vlm:
        try:
            import mlx_vlm  # noqa: check availability
            return MLXVLMCaptioner(config)
        except ImportError:
            logger.warning(
                "mlx-vlm not installed — falling back to Ollama. "
                "Install with: pip install mlx-vlm"
            )
            return OllamaCaptioner(config)
    else:
        return OllamaCaptioner(config)


def caption_frames(
    frames_meta: list[dict],
    config: Optional[CaptioningConfig] = None,
    output_dir: Optional[str] = None,
) -> list[dict]:
    """
    Generate captions for all extracted frames.

    Drives the captioner via ``caption_batch`` in chunks of ``config.batch_size``.
    For the MLX backend each chunk becomes one ``batch_generate`` call (shared
    KV cache, same-shape grouping). For the Ollama backend the chunk is a
    sequential loop, so batch_size has no throughput effect there.

    Adds a 'caption' field to each frame metadata dict.
    Optionally saves per-frame and combined caption JSON files to output_dir.
    """
    if config is None:
        config = CaptioningConfig()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    captioner = _get_captioner(config)
    backend_name = config.backend.value
    if config.backend == CaptionBackend.mlx_vlm:
        model_name = config.mlx_repo_id.split("/")[-1]
    else:
        model_name = config.ollama_model

    batch_size = max(1, int(config.batch_size))
    print(f"Captioning with {backend_name} ({model_name}) — batch_size={batch_size}")

    results: list[dict] = []
    pbar = tqdm(total=len(frames_meta), desc="Captioning frames")

    for chunk_start in range(0, len(frames_meta), batch_size):
        chunk = frames_meta[chunk_start : chunk_start + batch_size]
        image_paths = [f["path"] for f in chunk]

        try:
            captions = captioner.caption_batch(image_paths)
        except Exception as e:
            logger.error(
                f"Batch caption failed (frames "
                f"{chunk[0]['frame_id']}–{chunk[-1]['frame_id']}): {e}"
            )
            captions = [f"[Error captioning frame: {e}]"] * len(chunk)

        # Defensive: pad/truncate if the backend returned the wrong count
        if len(captions) != len(chunk):
            logger.warning(
                f"caption_batch returned {len(captions)} results for {len(chunk)} frames; "
                f"padding with error markers"
            )
            captions = (captions + ["[Error: missing caption]"] * len(chunk))[: len(chunk)]

        for frame, caption in zip(chunk, captions):
            enriched = {**frame, "caption": caption}
            results.append(enriched)

            if output_dir:
                caption_file = Path(output_dir) / f"caption_{frame['frame_id']:06d}.json"
                with open(caption_file, "w") as f:
                    json.dump(enriched, f, indent=2)

        pbar.update(len(chunk))

    pbar.close()

    # Save combined captions
    if output_dir:
        combined_file = Path(output_dir) / "all_captions.json"
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2)

    return results
