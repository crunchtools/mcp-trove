"""Tests for Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcp_trove_crunchtools.models import (
    GetChunksParams,
    IndexParams,
    ListParams,
    ReindexParams,
    RemoveParams,
    SearchParams,
    SimilarParams,
)


class TestSearchParams:
    def test_valid_minimal(self) -> None:
        params = SearchParams(query="test")
        assert params.query == "test"
        assert params.limit == 10
        assert params.path is None

    def test_valid_full(self) -> None:
        params = SearchParams(query="test query", path="/home/user", limit=50)
        assert params.query == "test query"
        assert params.path == "/home/user"
        assert params.limit == 50

    def test_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            SearchParams(query="")

    def test_limit_too_high(self) -> None:
        with pytest.raises(ValidationError):
            SearchParams(query="test", limit=200)

    def test_limit_too_low(self) -> None:
        with pytest.raises(ValidationError):
            SearchParams(query="test", limit=0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchParams(query="test", extra_field="bad")  # type: ignore[call-arg]


class TestSimilarParams:
    def test_valid(self) -> None:
        params = SimilarParams(file_path="/home/user/doc.txt")
        assert params.file_path == "/home/user/doc.txt"
        assert params.limit == 10

    def test_empty_path(self) -> None:
        with pytest.raises(ValidationError):
            SimilarParams(file_path="")


class TestIndexParams:
    def test_valid(self) -> None:
        params = IndexParams(path="/home/user/Documents")
        assert params.path == "/home/user/Documents"

    def test_empty_path(self) -> None:
        with pytest.raises(ValidationError):
            IndexParams(path="")


class TestReindexParams:
    def test_valid_with_path(self) -> None:
        params = ReindexParams(path="/home/user/Documents")
        assert params.path == "/home/user/Documents"

    def test_valid_without_path(self) -> None:
        params = ReindexParams()
        assert params.path is None


class TestRemoveParams:
    def test_valid(self) -> None:
        params = RemoveParams(path="/home/user/doc.txt")
        assert params.path == "/home/user/doc.txt"

    def test_empty_path(self) -> None:
        with pytest.raises(ValidationError):
            RemoveParams(path="")


class TestListParams:
    def test_valid_minimal(self) -> None:
        params = ListParams()
        assert params.path is None
        assert params.limit == 50
        assert params.offset == 0

    def test_valid_full(self) -> None:
        params = ListParams(path="/home/user", limit=100, offset=10)
        assert params.path == "/home/user"
        assert params.limit == 100
        assert params.offset == 10

    def test_negative_offset(self) -> None:
        with pytest.raises(ValidationError):
            ListParams(offset=-1)


class TestGetChunksParams:
    def test_valid(self) -> None:
        params = GetChunksParams(file_path="/home/user/doc.txt")
        assert params.file_path == "/home/user/doc.txt"
        assert params.limit == 50

    def test_empty_path(self) -> None:
        with pytest.raises(ValidationError):
            GetChunksParams(file_path="")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GetChunksParams(file_path="/test", unknown="bad")  # type: ignore[call-arg]
