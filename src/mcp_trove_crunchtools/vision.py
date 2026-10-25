"""Pluggable vision backends for image and video captioning."""

from __future__ import annotations

import base64
import contextvars
import io
import json
import logging
import os
import sqlite3
import threading
import urllib.request
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import SecretStr

from . import database, media
from .config import get_config
from .errors import ExtractionError, VisionDeferredError
from .media import AUDIO_CLIP_SECONDS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .media import MediaSample

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
    ".m4v": "video/mp4",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".wmv": "video/x-ms-wmv",
}

MEDIA_PROMPT = (
    "These are {frames} frames sampled evenly through a {kind}{audio}. For search "
    "indexing, describe in a few sentences: the subject, setting, people and "
    "activities, and any visible text. From the sound, say whether it is speech "
    "(its topic and language) or music (its genre and mood)."
)
AUDIO_PROMPT = (
    "This is a {seconds}-second clip from an audio file. For search indexing, say "
    "in a few sentences whether it is speech (its topic, language, kind of "
    "recording) or music (its genre, instruments and mood)."
)


class VisionBackend(Protocol):
    def caption(self, path: Path, file_type: str) -> str: ...


class GeminiBackend:
    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Gemini client (lazy, reused across calls)."""
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise ExtractionError(
                "GeminiBackend", "google-genai not installed (pip install google-genai)"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ExtractionError("GeminiBackend", "GEMINI_API_KEY not set")

        timeout_s = get_config().vision_timeout
        self._client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_s * 1000},
        )
        return self._client

    def caption(self, path: Path, file_type: str) -> str:
        _ = file_type
        try:
            from google.genai import types
        except ImportError as exc:
            raise ExtractionError(
                str(path), "google-genai not installed (pip install google-genai)"
            ) from exc

        client = self._get_client()
        mime = _get_mime(path)
        file_bytes = path.read_bytes()

        content = types.Content(
            parts=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime),
                types.Part.from_text(text=self._prompt),
            ]
        )
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=content,
            )
        except Exception as exc:
            raise ExtractionError(str(path), str(exc)) from exc
        if response.text:
            return str(response.text)
        raise ExtractionError(str(path), "Gemini returned empty response")


class OpenAIBackend:
    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI client (lazy, reused across calls)."""
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as exc:
            raise ExtractionError(
                "OpenAIBackend", "openai not installed (pip install openai)"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ExtractionError("OpenAIBackend", "OPENAI_API_KEY not set")

        self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def caption(self, path: Path, file_type: str) -> str:
        if file_type == "video":
            raise ExtractionError(str(path), "OpenAI vision does not support video files")

        client = self._get_client()
        mime = _get_mime(path)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        image_url = f"data:{mime};base64,{b64}"

        try:
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
                timeout=get_config().vision_timeout,
            )
        except Exception as exc:
            raise ExtractionError(str(path), str(exc)) from exc
        response_text = response.choices[0].message.content
        if response_text:
            return str(response_text)
        raise ExtractionError(str(path), "OpenAI returned empty response")


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Media is sent inline as a base64 data URL, so cap what we read into memory.
OPENROUTER_MAX_MEDIA_BYTES = 20 * 1024 * 1024
# A larger image is sent as a JPEG copy this many pixels on the long side.
OVERSIZE_IMAGE_SIDE = 2048
OVERSIZE_JPEG_QUALITY = 85


def _read_secret(name: str) -> SecretStr:
    """Read NAME_FILE if set (preferred), else NAME, stripped."""
    file_path = os.environ.get(f"{name}_FILE", "")
    if file_path:
        try:
            with open(file_path, encoding="utf-8") as f:
                return SecretStr(f.read().strip())
        except OSError as exc:
            raise ExtractionError("OpenRouterBackend", f"cannot read {name}_FILE: {exc}") from exc
    return SecretStr(os.environ.get(name, "").strip())


class OpenRouterBackend:
    """OpenRouter chat completions, restricted to zero-retention providers.

    Images go as image_url and videos as video_url base64 data URLs. The
    provider block keeps every request on ZDR endpoints that do not collect
    data. TROVE_VISION_FALLBACK_MODEL, when set, is tried when the first
    model is rate limited.
    """

    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI client pointed at OpenRouter."""
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as exc:
            raise ExtractionError(
                "OpenRouterBackend", "openai not installed (pip install openai)"
            ) from exc

        api_key = _read_secret("OPENROUTER_API_KEY")
        if not api_key.get_secret_value():
            raise ExtractionError("OpenRouterBackend", "OPENROUTER_API_KEY not set")

        self._client = openai.OpenAI(
            api_key=api_key.get_secret_value(), base_url=OPENROUTER_BASE_URL
        )
        return self._client

    def describe(self, path: Path, sample: MediaSample, kind: str) -> str:
        """Caption sampled frames and an audio clip together, in one request."""
        if sample.frames:
            text = MEDIA_PROMPT.format(
                frames=len(sample.frames),
                kind=kind,
                audio=", with a clip of its sound" if sample.audio else "",
            )
        elif sample.audio:
            text = AUDIO_PROMPT.format(seconds=AUDIO_CLIP_SECONDS)
        else:
            raise ExtractionError(str(path), "no video or audio stream could be decoded")
        self._get_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + _b64(frame)},
            }
            for frame in sample.frames
        )
        if sample.audio:
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": _b64(sample.audio), "format": "mp3"},
                }
            )
        return self._complete(path, content)

    def caption(self, path: Path, file_type: str) -> str:
        size = path.stat().st_size
        oversize = size > OPENROUTER_MAX_MEDIA_BYTES
        if oversize and file_type != "image":
            raise ExtractionError(
                str(path),
                f"{size} bytes exceeds the {OPENROUTER_MAX_MEDIA_BYTES} byte inline media limit",
            )
        self._get_client()  # a missing key fails before the file is read
        if oversize:
            mime, payload = "image/jpeg", _downscaled(path)
        else:
            mime, payload = _get_mime(path), path.read_bytes()
        data_url = f"data:{mime};base64," + _b64(payload)
        if file_type == "video":
            media = {"type": "video_url", "video_url": {"url": data_url}}
        else:
            media = {"type": "image_url", "image_url": {"url": data_url}}
        return self._complete(path, [{"type": "text", "text": self._prompt}, media])

    def _complete(self, path: Path, content: list[dict[str, Any]]) -> str:
        """One chat completion on a zero-retention provider, returning its text."""
        client = self._get_client()
        models = [self._model]
        fallback = get_config().vision_fallback_model
        if fallback and fallback != self._model:
            models.append(fallback)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                timeout=get_config().vision_timeout,
                extra_body={
                    "models": models,
                    "provider": {"zdr": True, "data_collection": "deny"},
                },
            )
        except Exception as exc:
            raise ExtractionError(str(path), str(exc)) from exc
        response_text = response.choices[0].message.content if response.choices else None
        if response_text:
            return str(response_text)
        raise ExtractionError(str(path), "OpenRouter returned empty response")


class OllamaBackend:
    def __init__(self, model: str, prompt: str) -> None:
        self._model = model
        self._prompt = prompt
        self._base_url = "http://localhost:11434"

    def caption(self, path: Path, file_type: str) -> str:
        if file_type == "video":
            raise ExtractionError(str(path), "Ollama vision does not support video files")

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": self._prompt,
                "images": [b64],
                "stream": False,
            }
        ).encode()

        ollama_url = f"{self._base_url}/api/generate"
        req = urllib.request.Request(
            ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
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


class _Budget:
    """One run's count of vision API calls, and whether the provider ran out of credit."""

    def __init__(self) -> None:
        self.calls = 0
        self.out_of_credit: str | None = None
        self.lock = threading.Lock()


# What a provider says when the account cannot pay for the call. Recorded as
# a per-file error, it would mark every remaining photo as a permanent
# failure, never retried; instead the run stops captioning and defers them.
_OUT_OF_CREDIT = (
    "error code: 402",
    "insufficient credit",
    "insufficient_quota",
    "payment required",
)


# Each run's budget travels in its context, which asyncio.to_thread copies
# into the worker threads, so overlapping runs (two MCP tool calls, say)
# count separately. Calls made outside any run share the default budget.
_current_budget: contextvars.ContextVar[_Budget | None] = contextvars.ContextVar(
    "trove_vision_budget", default=None
)
_default_budget = _Budget()
_cache_local = threading.local()
# Striped, so the lock set stays fixed however many files a run sees.
_CAPTION_LOCKS = tuple(threading.Lock() for _ in range(256))


def _run_budget() -> _Budget:
    """The current run's budget, or the shared default outside any run."""
    return _current_budget.get() or _default_budget


def reset_budget() -> None:
    """Start a new vision budget for the run in the current context."""
    _current_budget.set(_Budget())


def calls_made() -> int:
    """Vision API calls made under the current run's budget."""
    return _run_budget().calls


def out_of_credit() -> bool:
    """True when the provider refused a call this run for want of credit."""
    return _run_budget().out_of_credit is not None


def budget_spent() -> bool:
    """True once the current run can make no more vision calls: budget used, or no credit."""
    limit = get_config().vision_max_calls
    budget = _run_budget()
    return budget.out_of_credit is not None or (limit > 0 and budget.calls >= limit)


def _reserve_call(path: str) -> None:
    """Count one vision API call, or defer the file when the run's budget is spent."""
    limit = get_config().vision_max_calls
    budget = _run_budget()
    with budget.lock:
        if budget.out_of_credit is not None:
            raise VisionDeferredError(path, limit, budget.out_of_credit)
        if limit > 0 and budget.calls >= limit:
            raise VisionDeferredError(path, limit)
        budget.calls += 1


def _call(path: Path, make: Callable[[], str]) -> str:
    """Make one vision call; a refusal for want of credit defers the rest of the run."""
    try:
        return make()
    except ExtractionError as exc:
        message = str(exc).lower()
        if not any(marker in message for marker in _OUT_OF_CREDIT):
            raise
        budget = _run_budget()
        with budget.lock:
            budget.out_of_credit = "the vision provider is out of credit"
        logger.warning("Vision provider out of credit; deferring captions to the next run")
        raise VisionDeferredError(
            str(path), get_config().vision_max_calls, budget.out_of_credit
        ) from exc


def _cache_conn() -> sqlite3.Connection | None:
    """This thread's own connection to the caption cache, or None for :memory:.

    Extraction runs on worker threads, which cannot share the main thread's
    connection; cache lookups are single-row reads and writes, so each
    worker keeps its own, on the same file the main connection opened.
    """
    db_path = database.opened_path()
    if db_path is None or db_path == ":memory:":
        return None
    if getattr(_cache_local, "path", None) != db_path:
        _cache_local.conn = sqlite3.connect(db_path, timeout=30)
        _cache_local.path = db_path
    conn: sqlite3.Connection = _cache_local.conn
    return conn


def caption_cached(path: Path, file_type: str, checksum: str | None) -> str:
    """Caption a file, reusing the caption of any file with the same content.

    Captions are keyed by SHA-256, so a renamed or copied photo costs no API
    call. A new caption is saved as soon as it arrives, even if storing the
    file fails later, so a retry does not pay for it twice. Workers holding
    the same content wait on one lock stripe, so duplicates in one batch are
    captioned once.
    """
    backend = get_backend()
    if backend is None:
        raise ExtractionError(str(path), "No vision backend configured")
    return _cached(path, checksum, lambda: backend.caption(path, file_type))


def _cached(path: Path, checksum: str | None, make: Callable[[], str]) -> str:
    """The cached caption for checksum, or a new one from make(), saved at once."""
    conn = _cache_conn() if checksum else None
    if conn is None:
        _reserve_call(str(path))
        return _call(path, make)
    model = get_config().vision_model
    with _CAPTION_LOCKS[hash(checksum) % len(_CAPTION_LOCKS)]:
        # A caption from another model is not reused: changing
        # TROVE_VISION_MODEL re-captions files as they are re-indexed.
        row = conn.execute(
            "SELECT caption FROM captions WHERE sha256 = ? AND model = ?", (checksum, model)
        ).fetchone()
        if row:
            return str(row[0])
        _reserve_call(str(path))
        caption = _call(path, make)
        conn.execute(
            "INSERT OR REPLACE INTO captions (sha256, model, caption) VALUES (?, ?, ?)",
            (checksum, model, caption),
        )
        conn.commit()
    return caption


def describe_cached(path: Path, kind: str, checksum: str | None) -> str:
    """Caption a video or audio file from samples, reusing any cached caption.

    The cache is checked before anything is sampled. A cached caption of the
    whole video, from before sampling, is kept: it saw more. A backend that
    cannot take samples captions a video whole, as before, and cannot caption
    audio.
    """
    backend = get_backend()
    if backend is None:
        raise ExtractionError(str(path), "No vision backend configured")
    describe = getattr(backend, "describe", None)

    def make() -> str:
        if describe is not None:
            return str(describe(path, media.sample(path), kind))
        if kind == "video":
            return backend.caption(path, kind)
        raise ExtractionError(str(path), f"{get_config().vision_backend} cannot caption audio")

    return _cached(path, checksum, make)


def _downscaled(path: Path) -> bytes:
    """An image too big to send inline, as a JPEG at most OVERSIZE_IMAGE_SIDE a side.

    A 24 MB panorama or RAW-sized JPEG was a permanent failure; the model
    sees no more detail in it than in a 2048-pixel copy. draft() lets the
    JPEG decoder scale down as it reads, so memory follows the copy.
    """
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.draft("RGB", (OVERSIZE_IMAGE_SIDE, OVERSIZE_IMAGE_SIDE))
            image.thumbnail((OVERSIZE_IMAGE_SIDE, OVERSIZE_IMAGE_SIDE))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=OVERSIZE_JPEG_QUALITY)
    except Exception as exc:
        raise ExtractionError(str(path), f"cannot downscale for captioning: {exc}") from exc
    return buffer.getvalue()


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


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

    backends: dict[str, type[GeminiBackend | OpenAIBackend | OpenRouterBackend | OllamaBackend]] = {
        "gemini": GeminiBackend,
        "openai": OpenAIBackend,
        "openrouter": OpenRouterBackend,
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
