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
        """Remove <think>...</think> blocks from Qwen3.5 output (safety net)."""
        cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
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

    Adds a 'caption' field to each frame metadata dict.
    Optionally saves captions to JSON files in output_dir.
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

    print(f"Captioning with {backend_name} ({model_name})")

    results = []
    for frame in tqdm(frames_meta, desc="Captioning frames"):
        try:
            caption = captioner.caption(frame["path"])
        except Exception as e:
            caption = f"[Error captioning frame: {e}]"
            logger.error(f"Failed to caption frame {frame['frame_id']}: {e}")

        enriched = {**frame, "caption": caption}
        results.append(enriched)

        # Save individual caption
        if output_dir:
            caption_file = Path(output_dir) / f"caption_{frame['frame_id']:06d}.json"
            with open(caption_file, "w") as f:
                json.dump(enriched, f, indent=2)

    # Save combined captions
    if output_dir:
        combined_file = Path(output_dir) / "all_captions.json"
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2)

    return results
