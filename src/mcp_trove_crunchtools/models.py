"""Pydantic models for mcp-trove-crunchtools."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_PATH_LENGTH = 4096
MAX_QUERY_LENGTH = 2000
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 500


class SearchParams(BaseModel, extra="forbid"):
    """Parameters for hybrid search."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    path: str | None = Field(default=None, max_length=MAX_PATH_LENGTH)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT)


class SimilarParams(BaseModel, extra="forbid"):
    """Parameters for finding similar files."""

    file_path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT)


class IndexParams(BaseModel, extra="forbid"):
    """Parameters for indexing a path."""

    path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)


class ReindexParams(BaseModel, extra="forbid"):
    """Parameters for reindexing."""

    path: str | None = Field(default=None, max_length=MAX_PATH_LENGTH)


class RemoveParams(BaseModel, extra="forbid"):
    """Parameters for removing from index."""

    path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)


class ListParams(BaseModel, extra="forbid"):
    """Parameters for listing indexed files."""

    path: str | None = Field(default=None, max_length=MAX_PATH_LENGTH)
    limit: int = Field(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT)
    offset: int = Field(default=0, ge=0)


class GetChunksParams(BaseModel, extra="forbid"):
    """Parameters for getting chunks of a file."""

    file_path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)
    limit: int = Field(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT)
