"""Tests for configuration parsing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mcp_trove_crunchtools.config import (
    DEFAULT_MAX_TEXT_CHARS,
    _parse_max_text_chars,
    _parse_paths,
)


class TestParsePaths:
    def test_posix_separator(self) -> None:
        with patch("os.pathsep", ":"):
            assert _parse_paths("/home/user/docs:/data/files") == [
                "/home/user/docs",
                "/data/files",
            ]

    def test_windows_drive_letters_kept_intact(self) -> None:
        with patch("os.pathsep", ";"):
            assert _parse_paths(r"C:\Users\foo\Documents;D:\data") == [
                r"C:\Users\foo\Documents",
                r"D:\data",
            ]

    def test_empty_and_whitespace(self) -> None:
        assert _parse_paths("") == []
        assert _parse_paths("   ") == []

    def test_empty_segments_ignored(self) -> None:
        with patch("os.pathsep", ":"):
            assert _parse_paths("/one:: /two :") == ["/one", "/two"]


class TestParseMaxTextChars:
    def test_positive_value_kept(self) -> None:
        assert _parse_max_text_chars("2500") == 2500

    @pytest.mark.parametrize("raw", ["0", "-1"])
    def test_non_positive_falls_back_to_default(
        self, raw: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _parse_max_text_chars(raw) == DEFAULT_MAX_TEXT_CHARS
        assert "[WARNING] TROVE_MAX_TEXT_CHARS" in capsys.readouterr().err
