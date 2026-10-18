"""Tests for configuration parsing."""

from __future__ import annotations

from unittest.mock import patch

from mcp_trove_crunchtools.config import _parse_paths


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
