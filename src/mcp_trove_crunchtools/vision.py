"""Pluggable vision backends for image and video captioning."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Protocol

from .config import get_config
from .errors import ExtractionError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_MIMES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

_VIDEO_MIMES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}


class VisionBackend(Protocol):
    def caption(self, path: Path, file_type: str) -> str: ...


class GeminiBackend:
    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt

    def caption(self, path: Path, file_type: str) -> str:
        _ = file_type
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ExtractionError(
                str(path), "google-genai not installed (pip install google-genai)"
            ) from exc

        import os

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ExtractionError(str(path), "GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        mime = _get_mime(path)
        file_bytes = path.read_bytes()

        content = types.Content(
            parts=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime),
                types.Part.from_text(text=self._prompt),
            ]
        )
        response = client.models.generate_content(
            model=self._model,
            contents=content,
        )
        if response.text:
            return str(response.text)
        raise ExtractionError(str(path), "Gemini returned empty response")


class OpenAIBackend:
    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt

    def caption(self, path: Path, file_type: str) -> str:
        if file_type == "video":
            raise ExtractionError(str(path), "OpenAI vision does not support video files")

        try:
            import openai
        except ImportError as exc:
            raise ExtractionError(
                str(path), "openai not installed (pip install openai)"
            ) from exc

        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ExtractionError(str(path), "OPENAI_API_KEY not set")

        client = openai.OpenAI(api_key=api_key)
        mime = _get_mime(path)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        image_url = f"data:{mime};base64,{b64}"

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=300,
        )
        response_text = response.choices[0].message.content
        if response_text:
            return str(response_text)
        raise ExtractionError(str(path), "OpenAI returned empty response")


class OllamaBackend:
    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt
        self._base_url = "http://localhost:11434"

    def caption(self, path: Path, file_type: str) -> str:
        if file_type == "video":
            raise ExtractionError(str(path), "Ollama vision does not support video files")

        import json
        import urllib.request

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = json.dumps({
            "model": self._model,
            "prompt": self._prompt,
            "images": [b64],
            "stream": False,
        }).encode()

        ollama_url = f"{self._base_url}/api/generate"
        req = urllib.request.Request(
            ollama_url, data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                ollama_response = json.loads(resp.read().decode())
        except Exception as exc:
            raise ExtractionError(str(path), f"Ollama request failed: {exc}") from exc

        response_text = ollama_response.get("response", "")
        if response_text:
            return str(response_text)
        raise ExtractionError(str(path), "Ollama returned empty response")


def _get_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = _IMAGE_MIMES.get(suffix) or _VIDEO_MIMES.get(suffix)
    if mime:
        return mime
    raise ExtractionError(str(path), f"Unknown MIME type for extension: {suffix}")


_backend: VisionBackend | None = None


def get_backend() -> VisionBackend | None:
    """Get the configured vision backend, or None if vision is disabled."""
    global _backend
    if _backend is not None:
        return _backend

    cfg = get_config()
    if cfg.vision_backend == "none":
        return None

    backends: dict[str, type[GeminiBackend | OpenAIBackend | OllamaBackend]] = {
        "gemini": GeminiBackend,
        "openai": OpenAIBackend,
        "ollama": OllamaBackend,
    }
    backend_cls = backends.get(cfg.vision_backend)
    if backend_cls is None:
        logger.warning("Unknown vision backend: %s", cfg.vision_backend)
        return None

    _backend = backend_cls(cfg.vision_model, cfg.vision_prompt)
    return _backend


def reset_backend() -> None:
    """Reset the cached backend (for testing)."""
    global _backend
    _backend = None
