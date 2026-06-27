"""Small oMLX/OpenAI-compatible client used by ScreenLens.

The local oMLX server exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint, including vision inputs. This module keeps that dependency as a
plain HTTP adapter so the rest of the pipeline does not need the OpenAI SDK.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit

from .config import CaptioningConfig

logger = logging.getLogger("screenlens.omlx")


DEFAULT_OMLX_BASE_URL = "http://127.0.0.1:8000/v1"
_DOTENV_LOADED = False
_OMLX_KEY_PLACEHOLDERS = {
    "your-api-key",
    "your-api-key-here",
    "your-omlx-api-key",
    "your-omlx-api-key-here",
}
# Known text-only families. A vision marker (below) always overrides — so e.g.
# "MiniMax-VL" is still treated as vision even though "minimax" is listed here.
_KNOWN_TEXT_ONLY_PATTERNS = (
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-v4",
    "gpt-oss",
    "minimax-m1",
    "minimax-m2",
    "minimax-m3",
    "minimax-text",
    "kimi-k2",
    "nemotron",
    "glm-5-1",
)
_KNOWN_VISION_MARKERS = ("vl", "vision", "omni", "janus")
# Unified/multimodal model families whose names DON'T contain a vision marker
# but which do accept image input (verified June 2026). Matched on the
# normalized id (non-alphanumerics → '-'). VL/vision markers above still win.
_KNOWN_VISION_PATTERNS = (
    "gemma-4", "gemma-3",          # Gemma 3/4 are natively multimodal (OCR/doc/screen)
    "qwen3-6", "qwen3-5",          # Qwen3.5/3.6 are unified multimodal
    "qwen2-5-vl", "qwen3-vl",
    "pixtral", "internvl", "minicpm-v", "llava", "molmo", "kimi-vl",
)
# Draft/speculative-decode helpers — not standalone OCR models. "mtp" only
# counts as a standalone token (so "MTPLX-Optimized" — a real served model — is
# NOT flagged), via is_draft_model().
_DRAFT_MARKERS = ("dflash", "draft", "eagle")
_DRAFT_RE = re.compile(r"(^|-)mtp(-|$)")


def is_draft_model(model_id: str | None) -> bool:
    """True for speculative-decode draft models (not usable standalone)."""
    if not model_id:
        return False
    n = normalized_model_id(model_id)
    return any(m in n for m in _DRAFT_MARKERS) or bool(_DRAFT_RE.search(n))


def _load_dotenv_if_present() -> None:
    """Load simple KEY=VALUE entries from .env without overriding the shell."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        env_path = env_path.resolve()
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            os.environ.setdefault(key, value)


def _env_value(*names: str, ignore_placeholders: bool = False) -> str | None:
    _load_dotenv_if_present()
    for name in names:
        value = os.getenv(name)
        if not value:
            continue
        value = value.strip()
        if ignore_placeholders and value.lower() in _OMLX_KEY_PLACEHOLDERS:
            continue
        return value
    return None


def normalize_omlx_base_url(url: str) -> str:
    """Accept oMLX root/dashboard/API URLs and return the ``/v1`` base URL."""
    parsed = urlsplit(url)
    if parsed.path in ("", "/") or parsed.path.startswith("/admin"):
        return urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def resolve_omlx_base_url(config: CaptioningConfig) -> str:
    """Resolve oMLX base URL with Scriptorium-compatible env aliases."""
    env_url = _env_value("MLX_BASE_URL", "OMLX_BASE_URL")
    configured = config.omlx_base_url
    if configured and configured != DEFAULT_OMLX_BASE_URL:
        return normalize_omlx_base_url(configured)
    return normalize_omlx_base_url(env_url or configured or DEFAULT_OMLX_BASE_URL)


def resolve_omlx_api_key(config: CaptioningConfig) -> str | None:
    """Resolve oMLX API key with MLX_* and OMLX_* aliases."""
    return config.omlx_api_key or _env_value(
        "MLX_API_KEY",
        "OMLX_API_KEY",
        ignore_placeholders=True,
    )


def resolve_omlx_model(config: CaptioningConfig) -> str:
    """Resolve the model id to send to oMLX."""
    return (
        config.omlx_model
        or _env_value("MLX_MODEL", "OMLX_MODEL", "LLM_MODEL")
        or "default"
    )


# Default OCR (vision) model. Prefer whatever vision model is already served by
# your oMLX; override via OCR_MODEL env or ocr.model. Qwen3.x and Gemma-3/4 are
# unified multimodal and strong at dense-text/screen OCR.
RECOMMENDED_OCR_MODEL = "Qwen3.6-27B-bf16"


def resolve_ocr_model(config) -> str:
    """Resolve the OCR (vision) model id from an OCRConfig-like object."""
    return (
        getattr(config, "model", None)
        or _env_value("OCR_MODEL", "MLX_VISION_MODEL", "MLX_OCR_MODEL")
        or RECOMMENDED_OCR_MODEL
    )


def resolve_llm_model(config) -> str:
    """Resolve the text-LLM id from a ReconstructionConfig-like object."""
    return (
        getattr(config, "model", None)
        or _env_value("LLM_MODEL", "MLX_MODEL", "OMLX_MODEL")
        or "default"
    )


def list_models(base_url: str, api_key: str | None = None, timeout: float = 30.0) -> list[str]:
    """Return served model ids from the oMLX ``/v1/models`` endpoint."""
    base = normalize_omlx_base_url(base_url)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(f"{base}/models", headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except (HTTPError, URLError) as exc:  # pragma: no cover - network
        raise RuntimeError(f"Could not list oMLX models at {base}/models: {exc}") from exc
    items = data.get("data") or data.get("models") or []
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append(it.get("id") or it.get("name") or "")
        else:
            out.append(str(it))
    return [m for m in out if m]


def normalized_model_id(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower())


def is_known_vision_model(model_id: str | None) -> bool:
    """Return True if the id is a known vision/multimodal model."""
    if not model_id:
        return False
    n = normalized_model_id(model_id)
    if any(m in n for m in _KNOWN_VISION_MARKERS) or "ocr" in n:
        return True
    return any(p in n for p in _KNOWN_VISION_PATTERNS)


def is_known_text_only_model(model_id: str | None) -> bool:
    """Return True for served model ids that are known not to accept images."""
    if not model_id:
        return False
    if is_known_vision_model(model_id):
        return False
    normalized = normalized_model_id(model_id)
    return any(pattern in normalized for pattern in _KNOWN_TEXT_ONLY_PATTERNS)


def validate_omlx_vision_model(model_id: str) -> None:
    """Raise an actionable error if the selected model is known text-only."""
    if is_known_text_only_model(model_id):
        raise ValueError(
            f"{model_id} is a text-only oMLX model. ScreenLens captioning sends "
            "image inputs, so choose a vision-capable model such as a VL, vision, "
            "omni, or Janus model."
        )


def strip_thinking(text: str) -> str:
    """Remove Qwen/DeepSeek-style thinking blocks from final user-visible text.

    Handles three shapes:
      * complete ``<think>…</think>`` blocks,
      * a dangling ``</think>`` (opening tag was a prompt prefix) — keep what
        follows,
      * a dangling ``<think>`` with no close — generation was truncated mid-
        reasoning, so everything after it is thinking with no answer; drop it.
    """
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    elif "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def _image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "output_text"):
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


class OMLXClient:
    """Minimal chat client for an oMLX OpenAI-compatible server."""

    def __init__(self, config: CaptioningConfig):
        self.config = config
        self.base_url = resolve_omlx_base_url(config)
        self.model = resolve_omlx_model(config)
        self.api_key = resolve_omlx_api_key(config)
        self.timeout = config.omlx_timeout_seconds
        self._default_max_tokens = config.max_tokens
        self._default_temperature = config.temperature

    @classmethod
    def from_endpoint(
        cls,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float = 600.0,
        default_max_tokens: int = 4096,
        default_temperature: float = 0.0,
    ) -> "OMLXClient":
        """Build a client directly from endpoint params (no CaptioningConfig).

        Used by the verbatim OCR pass (vision model) and the reconstruction pass
        (text model), which keep their own config objects.
        """
        self = cls.__new__(cls)
        self.config = None
        self.base_url = normalize_omlx_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        return self

    def model_supports_vision(self) -> bool | None:
        """Best-effort: is this client's model vision-capable?

        Returns True/False from the name heuristic, or None if unknown. Used to
        fail loudly before sending images to a text-only model.
        """
        if is_known_text_only_model(self.model):
            return False
        if is_known_vision_model(self.model):
            return True
        return None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if images:
            validate_omlx_vision_model(self.model)

        user_content: str | list[dict[str, Any]]
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
                for path in images
            )
        else:
            user_content = user_prompt

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "temperature": temperature if temperature is not None else self._default_temperature,
            "stream": False,
        }
        # Pass-through sampler controls (repetition_penalty, no_repeat_ngram_size,
        # etc.). Unknown keys are ignored by most OpenAI-compatible MLX servers.
        if extra:
            payload.update({k: v for k, v in extra.items() if v is not None})
        return strip_thinking(self._post_chat(payload))

    def _post_chat(self, payload: dict[str, Any]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response = json.load(resp)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            hint = ""
            if exc.code == 401:
                hint = " Set MLX_API_KEY or OMLX_API_KEY, or pass --omlx-api-key."
            raise RuntimeError(
                f"oMLX chat completion failed with HTTP {exc.code}: {detail}{hint}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to oMLX at {self.base_url}. "
                "Start oMLX or pass --omlx-url."
            ) from exc

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"oMLX response contained no choices: {response}")

        first = choices[0]
        if isinstance(first, dict) and first.get("finish_reason") == "length":
            logger.warning(
                "oMLX truncated the response at max_tokens=%s (finish_reason=length). "
                "For a reasoning model this usually means it ran out of budget mid-"
                "thought and never reached the answer — disable thinking or raise "
                "max_tokens.", payload.get("max_tokens"),
            )
        if isinstance(first, dict):
            if "message" in first and isinstance(first["message"], dict):
                return _message_text(first["message"].get("content"))
            if "text" in first:
                return str(first["text"])
        return str(first)
