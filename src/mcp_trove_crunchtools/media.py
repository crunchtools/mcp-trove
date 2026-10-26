"""Sampling video and audio for captioning, without reading the whole file.

A video is described from a handful of frames spread across it and a short
clip of its sound, sent together in one vision request. Each frame comes
from the keyframe nearest its sample point, so a 2 GB file costs a few
ranged reads over the mount rather than a download, and every video costs
the same whatever its length (RT #1542). The opening seconds are not used
on their own: they are often black, a lens cap or an intro.

Container metadata (duration, recording time, the GPS a phone writes, and
an audio file's tags) is read for free.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import get_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

FRAME_COUNT = 10
FRAME_MAX_SIDE = 512  # pixels on the long side
FRAME_JPEG_QUALITY = 80
# Twice 8K UHD. One decoded frame of this size is about 200 MB; nothing a
# camera records comes close.
MAX_FRAME_PIXELS = 2 * 7680 * 4320
AUDIO_CLIP_SECONDS = 20
AUDIO_CLIP_START = 0.25  # of the duration: past any intro, before any outro
AUDIO_SAMPLE_RATE = 16000
AUDIO_BIT_RATE = 32000
# Bytes read from each end of a file for its fingerprint.
FINGERPRINT_SPAN = 1024 * 1024
# Packets read after a seek before giving up on a frame there.
MAX_PACKETS_PER_FRAME = 120

# ISO 6709, as phones write it: "+40.1234-082.9876+250.000/".
_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
_LOCATION_KEYS = ("location", "com.apple.quicktime.location.ISO6709", "location-eng")
_DATE_KEYS = ("creation_time", "com.apple.quicktime.creationdate", "date")
_TAG_KEYS = ("title", "artist", "album", "genre", "date", "comment")


@dataclass
class MediaSample:
    """What is sent for captioning, and what is indexed alongside the caption."""

    frames: list[bytes] = field(default_factory=list)  # JPEG
    audio: bytes | None = None  # MP3
    duration: float | None = None  # seconds
    recorded: str | None = None
    location: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """The metadata line for the index, or empty."""
        parts = []
        if self.duration:
            minutes, seconds = divmod(round(self.duration), 60)
            parts.append(f"Duration: {minutes}m{seconds:02d}s")
        if self.recorded:
            parts.append(f"Recorded: {self.recorded}")
        if self.location:
            parts.append(f"GPS: {self.location}")
        parts.extend(f"{key.title()}: {value}" for key, value in self.tags.items())
        return " | ".join(parts)


def fingerprint(path: Path) -> str:
    """A content key for a media file read from its size and three spans of it.

    Hashing a whole video would read all of it over the mount, which is the
    download sampling avoids. Size plus a megabyte from the start, middle
    and end tells apart any two real recordings (an edit that kept all three
    spans and the size would also have to keep every sampled frame), and
    still matches a renamed or copied file, so its caption is reused.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        digest.update(str(size).encode())
        for offset in sorted(
            {0, max(0, size // 2 - FINGERPRINT_SPAN // 2), max(0, size - FINGERPRINT_SPAN)}
        ):
            handle.seek(offset)
            digest.update(handle.read(FINGERPRINT_SPAN))
    return f"fp:{digest.hexdigest()}"


def probe(path: Path) -> MediaSample:
    """A video or audio file's metadata, read from its container header alone."""
    import av

    result = MediaSample()
    with av.open(str(path)) as container:
        metadata = {key.lower(): value for key, value in (container.metadata or {}).items()}
        if container.duration:
            result.duration = container.duration / av.time_base
    result.recorded = next((metadata[key] for key in _DATE_KEYS if metadata.get(key)), None)
    raw_location = next((metadata[key] for key in _LOCATION_KEYS if metadata.get(key)), "")
    location = _ISO6709.match(raw_location)
    if location and get_config().index_location:
        result.location = f"{float(location.group(1)):.4f}, {float(location.group(2)):.4f}"
    result.tags = {key: metadata[key] for key in _TAG_KEYS if metadata.get(key)}
    return result


def sample(path: Path) -> MediaSample:
    """A video or audio file's metadata, plus its sampled frames and audio clip.

    The frames and the audio are read in separate passes: seeking for frames
    leaves the demuxer at the end of the file.

    A song's embedded cover art is a one-frame "video" stream marked
    attached_pic; seeking in it fails with EPERM, which failed every song
    with art. It is not sampled: the song is described from its sound.

    Old camcorder AVIs keep no index for their sound, so seeking in it
    fails with EPERM too and leaves the demuxer unusable. The clip is then
    taken from the start of the file, opened again.
    """
    import av

    result = probe(path)
    with av.open(str(path)) as container:
        picture = next(
            (
                stream
                for stream in container.streams.video
                if not stream.disposition & av.stream.Disposition.attached_pic
            ),
            None,
        )
        if picture is not None:
            result.frames = _frames(container, picture, result.duration)
    for start in dict.fromkeys((_clip_start(result.duration), 0.0)):
        with av.open(str(path)) as container:
            if not container.streams.audio:
                break
            stream = container.streams.audio[0]
            if start and stream.time_base is not None and not _seek(container, stream, start):
                continue
            result.audio = _audio_clip(container, stream)
            break
    return result


def _seek(container: Any, stream: Any, seconds: float) -> bool:
    """Seek stream to the keyframe at or before seconds; False if refused."""
    import av

    try:
        container.seek(int(seconds / stream.time_base), stream=stream, backward=True)
    except av.FFmpegError as error:
        logger.info("Cannot seek in the %s stream: %s", stream.type, error)
        return False
    return True


def _frames(container: Any, stream: Any, duration: float | None) -> list[bytes]:
    """FRAME_COUNT frames at even points from 5% to 95% of the duration.

    Frames are decoded one at a time, so memory follows the picture size,
    not the file's: a stream larger than MAX_FRAME_PIXELS is not decoded.
    """
    width, height = stream.codec_context.width or 0, stream.codec_context.height or 0
    if width * height > MAX_FRAME_PIXELS:
        logger.warning("Not decoding %dx%d frames, over %d pixels", width, height, MAX_FRAME_PIXELS)
        return []
    if not duration or stream.time_base is None:
        first = _next_frame(container, stream)
        return [] if first is None else [_jpeg(first)]
    frames: list[bytes] = []
    for index in range(FRAME_COUNT):
        target = duration * (0.05 + 0.9 * index / max(1, FRAME_COUNT - 1))
        if not _seek(container, stream, target):
            break  # a refused seek leaves the demuxer unusable; keep what we have
        frame = _next_frame(container, stream)
        if frame is not None:
            frames.append(_jpeg(frame))
    return frames


def _next_frame(container: Any, stream: Any) -> Any:
    """The first frame the decoder produces from here, skipping packets it rejects.

    After a seek into an MPEG program stream or an Ogg Theora file, the
    first packets can reference frames before the seek point and fail to
    decode; they are skipped rather than failing the whole file.
    """
    return next(_decoded(container, stream, MAX_PACKETS_PER_FRAME), None)


def _decoded(container: Any, stream: Any, max_packets: int | None = None) -> Iterator[Any]:
    """Frames decoded from stream's packets, skipping any the decoder rejects."""
    import av

    rejected = 0
    try:
        for count, packet in enumerate(container.demux(stream)):
            if max_packets is not None and count >= max_packets:
                return
            try:
                yield from packet.decode()
            except av.FFmpegError:
                rejected += 1
    finally:
        if rejected:
            logger.debug("Skipped %d packets the %s decoder rejected", rejected, stream.type)


def _jpeg(frame: Any) -> bytes:
    image = frame.to_image()
    image.thumbnail((FRAME_MAX_SIDE, FRAME_MAX_SIDE))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=FRAME_JPEG_QUALITY)
    return buffer.getvalue()


def _clip_start(duration: float | None) -> float:
    """Where the audio clip starts: AUDIO_CLIP_START of a long file, else 0."""
    if duration and duration > AUDIO_CLIP_SECONDS * 2:
        return duration * AUDIO_CLIP_START
    return 0.0


def _audio_clip(container: Any, stream: Any) -> bytes | None:
    """AUDIO_CLIP_SECONDS of sound from where the demuxer stands, as mono MP3."""
    import av

    output = io.BytesIO()
    resampler = av.AudioResampler(format="s16p", layout="mono", rate=AUDIO_SAMPLE_RATE)
    with av.open(output, "w", format="mp3") as out:
        encoder = out.add_stream("mp3", rate=AUDIO_SAMPLE_RATE, layout="mono")
        encoder.bit_rate = AUDIO_BIT_RATE
        written = 0.0
        for frame in _decoded(container, stream):
            for converted in resampler.resample(frame):
                for packet in encoder.encode(converted):
                    out.mux(packet)
                written += converted.samples / AUDIO_SAMPLE_RATE
            if written >= AUDIO_CLIP_SECONDS:
                break
        for packet in encoder.encode(None):
            out.mux(packet)
    clip = output.getvalue()
    return clip or None
