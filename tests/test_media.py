"""Video and audio, captioned from samples instead of whole files (0.16.0)."""

from __future__ import annotations

import io
import shutil
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools import media, vision
from mcp_trove_crunchtools.config import get_config
from mcp_trove_crunchtools.extractor import MAX_FILE_SIZE, classify, extract_text
from mcp_trove_crunchtools.indexer import index_path_async
from mcp_trove_crunchtools.vision import OpenRouterBackend

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

RATE = 16000


def _make_clip(
    path: Path,
    seconds: int = 30,
    *,
    sound: bool = True,
    picture: bool = True,
    quiet_until: int = 0,
) -> Path:
    """A test clip: changing grey frames and a 440 Hz tone, with phone-style tags."""
    import av

    with av.open(str(path), "w") as container:
        container.metadata["creation_time"] = "2024-05-01T12:00:00Z"
        container.metadata["location"] = "+40.1234-082.9876/"
        container.metadata["title"] = "Basement training"
        # Every stream is declared before the first packet writes the header.
        video = container.add_stream("mpeg4", rate=10) if picture else None
        audio = (
            container.add_stream("aac" if picture else "mp3", rate=RATE, layout="mono")
            if sound
            else None
        )
        if video is not None:
            video.width, video.height, video.pix_fmt = 320, 180, "yuv420p"
            _write_frames(container, video, seconds)
        if audio is not None:
            _write_tone(container, audio, seconds, quiet_until)
    return path


def _write_frames(container: Any, stream: Any, seconds: int) -> None:
    import av
    import numpy as np

    for index in range(seconds * 10):
        image = np.full((180, 320, 3), (index * 3) % 255, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def _write_tone(container: Any, stream: Any, seconds: int, quiet_until: int) -> None:
    import av
    import numpy as np

    for start in range(0, seconds * RATE, 1024):
        loudness = 0 if start < quiet_until * RATE else 8000
        tone = np.sin(2 * np.pi * 440 * np.arange(start, start + 1024) / RATE) * loudness
        frame = av.AudioFrame.from_ndarray(
            tone.astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate, frame.pts = RATE, start
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


@pytest.fixture
def openrouter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """An OpenRouter backend whose API client is a mock answering every request."""
    backend = OpenRouterBackend("google/gemini-3.5-flash-lite", "describe this")
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="A person training in a basement gym."))
    ]
    backend._client = client
    monkeypatch.setattr(vision, "get_backend", lambda: backend)
    return client


def _request_content(client: MagicMock) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    return content


class TestSampling:
    def test_frames_audio_and_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config(), "index_location", True)
        sample = media.sample(_make_clip(tmp_path / "clip.mp4"))

        assert len(sample.frames) == media.FRAME_COUNT
        assert all(frame.startswith(b"\xff\xd8") for frame in sample.frames)  # JPEG
        assert sample.audio
        assert len(sample.audio) < 200_000
        assert sample.describe() == (
            "Duration: 0m30s | Recorded: 2024-05-01T12:00:00.000000Z | "
            "GPS: 40.1234, -82.9876 | Title: Basement training"
        )

    def test_audio_clip_starts_a_quarter_in(self, tmp_path: Path) -> None:
        """A 60 s clip silent for its first 14 s: a clip from 15 s is all tone."""
        import av
        import numpy as np

        clip = media.sample(_make_clip(tmp_path / "long.mp4", 60, quiet_until=14)).audio
        assert clip is not None
        with av.open(io.BytesIO(clip)) as container:
            first = np.concatenate(
                [frame.to_ndarray().ravel() for frame in container.decode(audio=0)][:8]
            )

        assert np.sqrt(np.mean(first.astype(np.float64) ** 2)) > 0.05

    def test_location_is_opt_in(self, tmp_path: Path) -> None:
        assert media.probe(_make_clip(tmp_path / "clip.mp4", 5)).location is None

    def test_silent_video_has_no_clip(self, tmp_path: Path) -> None:
        sample = media.sample(_make_clip(tmp_path / "silent.mp4", sound=False))

        assert len(sample.frames) == media.FRAME_COUNT
        assert sample.audio is None

    def test_audio_file_tags_and_clip(self, tmp_path: Path) -> None:
        sample = media.sample(_make_clip(tmp_path / "song.mp3", picture=False))

        assert sample.frames == []
        assert sample.audio
        assert sample.tags["title"] == "Basement training"

    def test_packets_the_decoder_rejects_are_skipped(self) -> None:
        """MPEG-PS and Theora files rejected their first packets after a seek."""
        import av

        bad, good = MagicMock(), MagicMock()
        bad.decode.side_effect = av.InvalidDataError(1094995529, "Invalid data")
        good.decode.return_value = ["frame"]
        container = MagicMock()
        container.demux.return_value = iter([bad, bad, good])

        assert media._next_frame(container, MagicMock()) == "frame"

    def test_fingerprint_follows_content_not_name(self, tmp_path: Path) -> None:
        clip = _make_clip(tmp_path / "clip.mp4")
        copy = tmp_path / "renamed.mp4"
        shutil.copy(clip, copy)
        other = _make_clip(tmp_path / "other.mp4", seconds=12)

        assert media.fingerprint(clip) == media.fingerprint(copy)
        assert media.fingerprint(clip) != media.fingerprint(other)
        assert media.fingerprint(clip).startswith("fp:")


class TestDefaults:
    def test_audio_is_opt_in(self, tmp_path: Path) -> None:
        with patch.object(vision, "get_backend", return_value=MagicMock()):
            assert classify(tmp_path / "voice memo.m4a") is None


class TestCaptioning:
    def test_one_request_carries_frames_and_sound(
        self, tmp_path: Path, openrouter: MagicMock
    ) -> None:
        text = extract_text(_make_clip(tmp_path / "clip.mp4"), "fp:abc")

        assert openrouter.chat.completions.create.call_count == 1
        kinds = [part["type"] for part in _request_content(openrouter)]
        assert kinds == ["text"] + ["image_url"] * media.FRAME_COUNT + ["input_audio"]
        assert "[Video] Duration: 0m30s" in text
        assert "[Caption] A person training in a basement gym." in text
        assert vision.calls_made() == 1

    def test_audio_file_sends_only_sound(
        self, tmp_path: Path, openrouter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = get_config()
        monkeypatch.setattr(config, "index_types", config.index_types | {"audio"})
        extract_text(_make_clip(tmp_path / "song.mp3", picture=False), "fp:abc")

        kinds = [part["type"] for part in _request_content(openrouter)]
        assert kinds == ["text", "input_audio"]

    def test_cached_caption_skips_sampling(
        self, tmp_path: Path, openrouter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = tmp_path / "trove.db"
        db.get_db(str(database))
        clip = _make_clip(tmp_path / "clip.mp4")
        extract_text(clip, "fp:same")
        sampled = MagicMock(wraps=media.sample)
        monkeypatch.setattr(media, "sample", sampled)

        extract_text(clip, "fp:same")

        assert openrouter.chat.completions.create.call_count == 1
        sampled.assert_not_called()


class TestIndexing:
    def test_big_video_is_still_a_video(self, tmp_path: Path) -> None:
        with patch.object(vision, "get_backend", return_value=MagicMock()):
            assert classify(tmp_path / "movie.mp4", MAX_FILE_SIZE * 40) == "video"

    async def test_video_is_fingerprinted_not_hashed(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, openrouter: MagicMock
    ) -> None:
        _make_clip(tmp_path / "clip.mp4")

        results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["indexed"]
        row = db.query_one("SELECT file_type, checksum FROM files")
        assert row is not None
        assert row["file_type"] == "video"
        assert row["checksum"].startswith("fp:")

    def test_migration_releases_old_video_failures(self, tmp_path: Path) -> None:
        database = tmp_path / "trove.db"
        db.get_db(str(database))
        db.insert_error(
            None, "/v/clip.MP4", "exceeds the inline media limit", "permanent", (9, 1.0)
        )
        db.insert_error(None, "/d/scan.pdf", "broken", "permanent", (9, 1.0))
        db.get_db().execute("PRAGMA user_version = 2")
        db.get_db().commit()
        db.get_db().close()
        db._db = None

        db.get_db(str(database))

        assert db.known_failures() == {("/d/scan.pdf", 9, 1.0)}
