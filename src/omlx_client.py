"""Small oMLX/OpenAI-compatible client used by ScreenLens.

The local oMLX server exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint, including vision inputs. This module keeps that dependency as a
plain HTTP adapter so the rest of the pipeline does not need the OpenAI SDK.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit

from .config import CaptioningConfig


DEFAULT_OMLX_BASE_URL = "http://127.0.0.1:8000/v1"
_DOTENV_LOADED = False
_OMLX_KEY_PLACEHOLDERS = {
    "your-api-key",
    "your-api-key-here",
    "your-omlx-api-key",
    "your-omlx-api-key-here",
}


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


def strip_thinking(text: str) -> str:
    """Remove Qwen/DeepSeek-style thinking blocks from final user-visible text."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
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

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
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
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": False,
        }
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
        if isinstance(first, dict):
            if "message" in first and isinstance(first["message"], dict):
                return _message_text(first["message"].get("content"))
            if "text" in first:
                return str(first["text"])
        return str(first)
