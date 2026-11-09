"""Tests for vision backends."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

from mcp_trove_crunchtools import vision
from mcp_trove_crunchtools.errors import ExtractionError
from mcp_trove_crunchtools.vision import (
    OPENROUTER_MAX_MEDIA_BYTES,
    OVERSIZE_IMAGE_SIDE,
    GeminiBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenRouterBackend,
    _get_mime,
    get_backend,
    reset_backend,
)


class TestGetMime:
    def test_jpeg(self) -> None:
        assert _get_mime(Path("photo.jpg")) == "image/jpeg"
        assert _get_mime(Path("photo.jpeg")) == "image/jpeg"

    def test_png(self) -> None:
        assert _get_mime(Path("photo.png")) == "image/png"

    def test_video(self) -> None:
        assert _get_mime(Path("clip.mp4")) == "video/mp4"
        assert _get_mime(Path("clip.mov")) == "video/quicktime"

    def test_unknown(self) -> None:
        with pytest.raises(ExtractionError, match="Unknown MIME type"):
            _get_mime(Path("file.xyz"))


class TestGetBackend:
    def test_none_backend(self) -> None:
        """Default config returns None (vision disabled)."""
        reset_backend()
        assert get_backend() is None

    def test_gemini_backend(self) -> None:
        reset_backend()
        with patch.dict(os.environ, {"TROVE_VISION_BACKEND": "gemini"}):
            from mcp_trove_crunchtools import config as config_mod

            config_mod._config = None
            backend = get_backend()
            assert isinstance(backend, GeminiBackend)

    def test_openai_backend(self) -> None:
        reset_backend()
        with patch.dict(os.environ, {"TROVE_VISION_BACKEND": "openai"}):
            from mcp_trove_crunchtools import config as config_mod

            config_mod._config = None
            backend = get_backend()
            assert isinstance(backend, OpenAIBackend)

    def test_ollama_backend(self) -> None:
        reset_backend()
        with patch.dict(os.environ, {"TROVE_VISION_BACKEND": "ollama"}):
            from mcp_trove_crunchtools import config as config_mod

            config_mod._config = None
            backend = get_backend()
            assert isinstance(backend, OllamaBackend)

    def test_unknown_backend(self) -> None:
        reset_backend()
        with patch.dict(os.environ, {"TROVE_VISION_BACKEND": "unknown"}):
            from mcp_trove_crunchtools import config as config_mod

            config_mod._config = None
            assert get_backend() is None


class TestGeminiBackend:
    def _make_mock_genai(self) -> MagicMock:
        """Create a mock google.genai module with types submodule."""
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_genai.types = mock_types
        return mock_genai

    def test_caption_no_api_key(self) -> None:
        backend = GeminiBackend("gemini-2.5-flash", "describe this")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            path = Path(f.name)

        mock_genai = self._make_mock_genai()
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False),
            patch.dict(
                sys.modules,
                {
                    "google": MagicMock(),
                    "google.genai": mock_genai,
                    "google.genai.types": mock_genai.types,
                },
            ),
            pytest.raises(ExtractionError, match="GEMINI_API_KEY not set"),
        ):
            backend.caption(path, "image")
        path.unlink()

    def test_caption_success(self) -> None:
        backend = GeminiBackend("gemini-2.5-flash", "describe this")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            path = Path(f.name)

        mock_response = MagicMock()
        mock_response.text = "A beautiful sunset over Brussels"
        mock_genai = self._make_mock_genai()
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(
                sys.modules,
                {
                    "google": mock_google,
                    "google.genai": mock_genai,
                    "google.genai.types": mock_genai.types,
                },
            ),
        ):
            result = backend.caption(path, "image")

        assert result == "A beautiful sunset over Brussels"
        path.unlink()

    def test_caption_empty_response(self) -> None:
        backend = GeminiBackend("gemini-2.5-flash", "describe this")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            path = Path(f.name)

        mock_response = MagicMock()
        mock_response.text = ""
        mock_genai = self._make_mock_genai()
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(
                sys.modules,
                {
                    "google": mock_google,
                    "google.genai": mock_genai,
                    "google.genai.types": mock_genai.types,
                },
            ),
            pytest.raises(ExtractionError, match="empty response"),
        ):
            backend.caption(path, "image")
        path.unlink()


class TestOpenAIBackend:
    def test_video_not_supported(self) -> None:
        backend = OpenAIBackend("gpt-4o-mini", "describe this")
        with pytest.raises(ExtractionError, match="does not support video"):
            backend.caption(Path("clip.mp4"), "video")

    def test_caption_no_api_key(self) -> None:
        backend = OpenAIBackend("gpt-4o-mini", "describe this")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            path = Path(f.name)

        mock_openai = MagicMock()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            patch.dict(sys.modules, {"openai": mock_openai}),
            pytest.raises(ExtractionError, match="OPENAI_API_KEY not set"),
        ):
            backend.caption(path, "image")
        path.unlink()


class TestOllamaBackend:
    def test_video_not_supported(self) -> None:
        backend = OllamaBackend("llava", "describe this")
        with pytest.raises(ExtractionError, match="does not support video"):
            backend.caption(Path("clip.mp4"), "video")


class TestExifExtraction:
    """Test EXIF metadata extraction from images."""

    def test_extract_exif_no_data(self) -> None:
        """Returns empty string for images without EXIF."""
        from mcp_trove_crunchtools.extractor import _extract_exif

        mock_img = MagicMock()
        mock_img.getexif.return_value = {}
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)

        with patch("PIL.Image.open", return_value=mock_img):
            result = _extract_exif(Path("test.jpg"))
        assert result == ""

    def test_dms_to_decimal(self) -> None:
        """Test DMS to decimal conversion."""
        from mcp_trove_crunchtools.extractor import _dms_to_decimal

        # 50 degrees, 50 minutes, 48 seconds = 50.8467 degrees
        result = _dms_to_decimal((50.0, 50.0, 48.0))
        assert abs(result - 50.8467) < 0.001


class TestOpenRouterBackend:
    def test_backend_selected(self) -> None:
        reset_backend()
        with patch.dict(os.environ, {"TROVE_VISION_BACKEND": "openrouter"}):
            from mcp_trove_crunchtools import config as config_mod

            config_mod._config = None
            backend = get_backend()
            assert isinstance(backend, OpenRouterBackend)
            assert backend._model == "google/gemini-3.1-flash-lite"
        reset_backend()
        config_mod._config = None

    def test_caption_no_api_key(self) -> None:
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        mock_openai = MagicMock()
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "OPENROUTER_API_KEY_FILE": ""}),
            patch.dict(sys.modules, {"openai": mock_openai}),
            pytest.raises(ExtractionError, match="OPENROUTER_API_KEY not set"),
        ):
            backend.caption(Path(__file__), "image")

    def test_key_file_takes_precedence(self, tmp_path: Path) -> None:
        key_file = tmp_path / "key"
        key_file.write_text("sk-or-file\n")
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        mock_openai = MagicMock()
        with (
            patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "sk-or-env",
                    "OPENROUTER_API_KEY_FILE": str(key_file),
                },
            ),
            patch.dict(sys.modules, {"openai": mock_openai}),
        ):
            backend._get_client()
        mock_openai.OpenAI.assert_called_once_with(
            api_key="sk-or-file", base_url="https://openrouter.ai/api/v1"
        )

    def test_video_sent_as_video_url_with_zdr(self, tmp_path: Path) -> None:
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="a river at dusk"))
        ]
        backend._client = client
        assert backend.caption(clip, "video") == "a river at dusk"
        kwargs = client.chat.completions.create.call_args.kwargs
        media = kwargs["messages"][0]["content"][1]
        assert media["type"] == "video_url"
        assert media["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert kwargs["extra_body"]["provider"] == {"zdr": True, "data_collection": "deny"}
        assert kwargs["extra_body"]["models"] == ["google/gemini-3.1-flash-lite"]

    def test_empty_response(self, tmp_path: Path) -> None:
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG")
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = []
        backend._client = client
        with pytest.raises(ExtractionError, match="empty response"):
            backend.caption(img, "image")

    def test_image_sent_as_image_url(self, tmp_path: Path) -> None:
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG")
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="a red barn"))
        ]
        backend._client = client
        assert backend.caption(img, "image") == "a red barn"
        media = client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]
        assert media["type"] == "image_url"
        assert media["image_url"]["url"].startswith("data:image/png;base64,")

    def test_api_error_becomes_extraction_error(self, tmp_path: Path) -> None:
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG")
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("HTTP 429")
        backend._client = client
        with pytest.raises(ExtractionError, match="HTTP 429"):
            backend.caption(img, "image")

    def test_oversized_media_rejected_before_read(self, tmp_path: Path) -> None:
        clip = tmp_path / "big.mp4"
        with clip.open("wb") as f:
            f.truncate(OPENROUTER_MAX_MEDIA_BYTES + 1)
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        backend._client = MagicMock()
        with pytest.raises(ExtractionError, match="inline media limit"):
            backend.caption(clip, "video")
        backend._client.chat.completions.create.assert_not_called()

    def test_oversized_image_sent_downscaled(self, tmp_path: Path) -> None:
        """A photo over the inline limit is captioned from a JPEG copy, not failed."""
        import numpy as np
        from PIL import Image

        noise = np.random.default_rng(1).integers(0, 255, (3000, 3000, 3), dtype=np.uint8)
        img = tmp_path / "pano.png"
        Image.fromarray(noise).save(img, compress_level=0)
        assert img.stat().st_size > OPENROUTER_MAX_MEDIA_BYTES
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="a wide valley"))
        ]
        backend._client = client

        assert backend.caption(img, "image") == "a wide valley"
        url = client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1][
            "image_url"
        ]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert max(sent.size) == OVERSIZE_IMAGE_SIDE

    def test_oversized_corrupt_image_is_an_extraction_error(self, tmp_path: Path) -> None:
        img = tmp_path / "broken.jpg"
        with img.open("wb") as f:
            f.write(b"\xff\xd8\xff\xe0 not really a jpeg")
            f.truncate(OPENROUTER_MAX_MEDIA_BYTES + 1)
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        backend._client = MagicMock()

        with pytest.raises(ExtractionError, match="cannot downscale"):
            backend.caption(img, "image")
        backend._client.chat.completions.create.assert_not_called()

    def test_unreadable_key_file(self, tmp_path: Path) -> None:
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY_FILE": str(tmp_path / "missing")}),
            patch.dict(sys.modules, {"openai": MagicMock()}),
            pytest.raises(ExtractionError, match="cannot read OPENROUTER_API_KEY_FILE"),
        ):
            backend._get_client()

    def test_openai_not_installed(self) -> None:
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe this")
        with (
            patch.dict(sys.modules, {"openai": None}),
            pytest.raises(ExtractionError, match="openai not installed"),
        ):
            backend._get_client()


class TestOllamaHttpContract:
    """The outbound HTTP contract (constitution v1.36.0, profile Layer 3).

    A real `httpx.Client` over a `MockTransport`: the agent string is a header
    the client adds and the ceiling is enforced while the client reads, so
    mocking the client's own method would mock away both.
    """

    @staticmethod
    @contextmanager
    def _serving(handler: Callable[[httpx.Request], httpx.Response]) -> Iterator[None]:
        real = httpx.Client

        def factory(**kwargs: Any) -> httpx.Client:
            kwargs.pop("transport", None)
            return real(transport=httpx.MockTransport(handler), **kwargs)

        with patch.object(httpx, "Client", factory):
            yield

    @staticmethod
    def _image(tmp_path: Path) -> Path:
        path = tmp_path / "photo.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
        return path

    def test_the_request_names_the_product_and_version(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["User-Agent"])
            return httpx.Response(200, json={"response": "a cat"})

        with self._serving(handler):
            caption = OllamaBackend("llava", "describe").caption(self._image(tmp_path), "image")

        assert caption == "a cat"
        expected = (
            f"mcp-trove-crunchtools/{version('mcp-trove-crunchtools')} "
            "(+https://github.com/crunchtools/mcp-trove)"
        )
        assert seen == [expected]
        assert "python-httpx" not in seen[0]

    def test_a_chunked_body_over_the_cap_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`.read()` with no argument let whatever was on port 11434 decide."""
        monkeypatch.setattr(vision, "MAX_RESPONSE_SIZE", 4096)

        def chunks() -> Iterator[bytes]:
            for _ in range(16):
                yield b"x" * 1024

        def handler(_request: httpx.Request) -> httpx.Response:
            response = httpx.Response(
                200, headers={"content-type": "application/json"}, content=chunks()
            )
            assert "content-length" not in response.headers
            return response

        with self._serving(handler), pytest.raises(ExtractionError) as caught:
            OllamaBackend("llava", "describe").caption(self._image(tmp_path), "image")

        assert "over the 4096 byte limit" in str(caught.value)

    def test_a_non_json_answer_says_what_came_back(self, tmp_path: Path) -> None:
        """Something else listening on 11434 is the case this has to name."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text="<title>Grafana</title>"
            )

        with self._serving(handler), pytest.raises(ExtractionError) as caught:
            OllamaBackend("llava", "describe").caption(self._image(tmp_path), "image")

        message = str(caught.value)
        assert "Grafana" in message
        assert "text/html" in message
        assert "localhost:11434/api/generate" in message

    def test_an_error_status_quotes_the_body(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model llava not found"})

        with self._serving(handler), pytest.raises(ExtractionError) as caught:
            OllamaBackend("llava", "describe").caption(self._image(tmp_path), "image")

        message = str(caught.value)
        assert "model llava not found" in message
        assert "HTTP 404" in message
