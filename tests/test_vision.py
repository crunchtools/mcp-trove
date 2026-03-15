"""Tests for vision backends."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_trove_crunchtools.errors import ExtractionError
from mcp_trove_crunchtools.vision import (
    GeminiBackend,
    OllamaBackend,
    OpenAIBackend,
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
            patch.dict(sys.modules, {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_genai.types,
            }),
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
            patch.dict(sys.modules, {
                "google": mock_google,
                "google.genai": mock_genai,
                "google.genai.types": mock_genai.types,
            }),
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
            patch.dict(sys.modules, {
                "google": mock_google,
                "google.genai": mock_genai,
                "google.genai.types": mock_genai.types,
            }),
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
