"""Test fixtures for mcp-trove-crunchtools."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from mcp_trove_crunchtools import config as config_mod
from mcp_trove_crunchtools import database as db_mod
from mcp_trove_crunchtools import embedder as embedder_mod
from mcp_trove_crunchtools import indexer as indexer_mod
from mcp_trove_crunchtools import vision as vision_mod
from mcp_trove_crunchtools.tools import search as search_mod

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
def _index_other_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests see the allowlist alone; name-only entries are tested on their own."""
    monkeypatch.setenv("TROVE_INDEX_OTHER", "skip")


@pytest.fixture(autouse=True)
def _reset_singletons() -> Generator[None]:
    """Reset config, database, and embedder singletons between tests."""
    config_mod._config = None
    db_mod._db = None
    db_mod._tx_depth = 0
    embedder_mod._model = None
    vision_mod._backend = None
    vision_mod.reset_budget()
    yield
    if db_mod._db is not None:
        db_mod._db.close()
    db_mod._db = None
    config_mod._config = None
    embedder_mod._model = None
    vision_mod._backend = None


@pytest.fixture(autouse=True)
def _mock_embedder() -> Generator[None]:
    """Mock the embedder functions to avoid loading real models in tests."""
    with (
        patch.object(indexer_mod, "embed_texts", side_effect=_mock_embed_texts),
        patch.object(search_mod, "embed_query", side_effect=_mock_embed_query),
    ):
        yield


@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with schema."""
    return db_mod.get_db(":memory:")


class SdkCalls:
    """The requests an openai-SDK-backed backend actually sent.

    The suite used to script `chat.completions.create` on a `MagicMock`, which
    mocked away the thing under test: the agent string is a header the client
    adds and the ceiling is enforced while the body is read, so a mock standing
    in for the call could see neither. A `MockTransport` under a client the
    production code builds itself sees both -- and these accessors read the
    request off the wire, which is a stronger assertion than reading back the
    kwargs a mock was handed.
    """

    def __init__(self) -> None:
        self.requests: list[Any] = []

    @property
    def count(self) -> int:
        return len(self.requests)

    def payload(self, index: int = -1) -> dict[str, Any]:
        """The JSON body of a recorded request."""
        return cast("dict[str, Any]", json.loads(self.requests[index].content))

    def content(self, index: int = -1) -> list[dict[str, Any]]:
        """The message content list of a recorded chat completion."""
        return cast("list[dict[str, Any]]", self.payload(index)["messages"][0]["content"])

    def header(self, name: str, index: int = -1) -> str | None:
        return cast("str | None", self.requests[index].headers.get(name))


def caption_reply(text: str = "a caption") -> Any:
    """A chat-completions reply carrying `text`."""
    import httpx2

    return httpx2.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        headers={"content-type": "application/json"},
    )


@contextmanager
def openai_sdk(reply: Any) -> Generator[SdkCalls, None, None]:
    """Put a MockTransport under every client the openai SDK builds.

    `reply` is either a response or a callable taking the request. The patch is
    on `openai.OpenAI`, so the backend still constructs its own client with its
    own headers; only the transport comes from the test.
    """
    import httpx2
    import openai

    calls = SdkCalls()
    real = openai.OpenAI

    def handler(request: Any) -> Any:
        calls.requests.append(request)
        return reply(request) if callable(reply) else reply

    def factory(**kwargs: Any) -> Any:
        # The transport goes INSIDE the client production code asked for, so
        # the response hook it installs is still the one running. Replacing the
        # client would take the hook off with it, and the error path is bounded
        # by nothing else.
        given = kwargs.pop("http_client", None)
        hooks = getattr(given, "event_hooks", {}) if given is not None else {}
        kwargs["http_client"] = httpx2.Client(
            transport=httpx2.MockTransport(handler), event_hooks=hooks
        )
        return real(**kwargs)

    with patch.object(openai, "OpenAI", factory):
        yield calls


@contextmanager
def openrouter_backend(
    reply: Any, model: str = "google/gemini-3.1-flash-lite", prompt: str = "describe this"
) -> Generator[tuple[Any, SdkCalls], None, None]:
    """An OpenRouter backend that builds its own client, over a mock transport."""
    from mcp_trove_crunchtools.vision import OpenRouterBackend

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}),  # gatehouse: allow-secret
        openai_sdk(reply) as calls,
    ):
        yield OpenRouterBackend(model, prompt), calls
