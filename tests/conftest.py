"""Test fixtures for mcp-trove-crunchtools."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from mcp_trove_crunchtools import config as config_mod
from mcp_trove_crunchtools import database as db_mod
from mcp_trove_crunchtools import embedder as embedder_mod

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator

MOCK_DIMS = 384


def _mock_embedding(seed: int = 0) -> list[float]:
    """Generate a deterministic mock embedding vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(MOCK_DIMS).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _mock_embed_texts(texts: list[str]) -> list[list[float]]:
    """Mock embed_texts that returns deterministic vectors."""
    return [_mock_embedding(hash(t) % 10000) for t in texts]


def _mock_embed_query(text: str) -> list[float]:
    """Mock embed_query that returns a deterministic vector."""
    return _mock_embedding(hash(text) % 10000)


@pytest.fixture(autouse=True)
def _reset_singletons() -> Generator[None]:
    """Reset config, database, and embedder singletons between tests."""
    config_mod._config = None
    db_mod._db = None
    embedder_mod._model = None
    yield
    if db_mod._db is not None:
        db_mod._db.close()
    db_mod._db = None
    config_mod._config = None
    embedder_mod._model = None


@pytest.fixture(autouse=True)
def _mock_embedder() -> Generator[None]:
    """Mock the embedder functions to avoid loading real models in tests."""
    with (
        patch.object(embedder_mod, "embed_texts", side_effect=_mock_embed_texts),
        patch.object(embedder_mod, "embed_query", side_effect=_mock_embed_query),
    ):
        yield


@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with schema."""
    return db_mod.get_db(":memory:")
