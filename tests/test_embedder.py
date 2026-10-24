"""Tests for the embedding wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from mcp_trove_crunchtools import embedder


class TestEncodeBatchSize:
    """fastembed's default batch of 256 long chunks OOM-killed the indexer
    (RT #1542), so embed_texts passes a small batch size."""

    def test_embed_texts_passes_small_batch(self) -> None:
        model = MagicMock()
        model.embed.return_value = iter([np.zeros(3), np.ones(3)])

        with patch.object(embedder, "get_model", return_value=model):
            vectors = embedder.embed_texts(["a", "b"])

        model.embed.assert_called_once_with(["a", "b"], batch_size=embedder.ENCODE_BATCH_SIZE)
        assert embedder.ENCODE_BATCH_SIZE <= 16
        assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
