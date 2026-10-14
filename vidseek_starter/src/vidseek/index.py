"""SQLite-backed local index for video embeddings."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .encoder import VideoEncoder


_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_video_segments_source ON video_segments(source_path);
"""


class VideoIndex:
    """Persist embeddings locally and retrieve the most relevant video segments.

    Args:
        database_path: SQLite path. Use ``:memory:`` for a temporary in-memory index.
        encoder: Optional preconfigured encoder, useful for shared model instances/testing.
    """

    def __init__(
        self,
        database_path: str | Path = "./vidseek.sqlite3",
        encoder: VideoEncoder | None = None,
    ) -> None:
        self.database_path = str(database_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.database_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:")
        else:
            Path(self.database_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self.encoder = encoder or VideoEncoder()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection and close file-backed connections deterministically."""
        is_memory = self._memory_connection is not None
        connection = self._memory_connection if is_memory else sqlite3.connect(self.database_path)
        assert connection is not None
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            if not is_memory:
                connection.close()

    @staticmethod
    def _vector_blob(embedding: Any) -> tuple[bytes, int]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a non-empty finite numeric vector")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("embedding must not be a zero vector")
        vector = vector / norm
        return vector.tobytes(), int(vector.size)

    @staticmethod
    def _metadata_json(metadata: dict[str, Any] | None) -> str:
        try:
            return json.dumps(metadata or {}, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc

    def add_embedding(
        self,
        source_path: str | Path,
        embedding: Any,
        *,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert one precomputed embedding (advanced/integration API)."""
        if start_seconds < 0 or (end_seconds is not None and end_seconds < start_seconds):
            raise ValueError("invalid segment timestamps")
        path = str(Path(source_path).expanduser().resolve())
        blob, dimensions = self._vector_blob(embedding)
        metadata_json = self._metadata_json(metadata)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO video_segments "
                "(source_path, start_seconds, end_seconds, embedding, dimensions, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (path, float(start_seconds), end_seconds, blob, dimensions, metadata_json),
            )

    def add_video(
        self,
        video_path: str | Path,
        *,
        segment_seconds: float | None = 20.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Embed and index a video; by default create timestamped 20-second chunks.

        Requires ``ffmpeg`` and ``ffprobe`` on PATH when ``segment_seconds`` is set.
        Set ``segment_seconds=None`` to create one embedding for the complete video.
        Existing entries for the same resolved path are replaced only after the new
        embeddings have all been computed successfully.

        Returns the number of indexed segments.
        """
        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        if segment_seconds is not None and segment_seconds <= 0:
            raise ValueError("segment_seconds must be greater than 0, or None")

        if segment_seconds is None:
            duration = _probe_duration(source, required=False)
            duration = float(duration) if duration is not None else None
            segments = [(source, 0.0, duration)]
            embeddings = [self.encoder.embed_video(source)]
        else:
            duration = _probe_duration(source, required=True)
            if duration <= 0:
                raise ValueError(f"Could not determine a positive video duration: {source}")
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError(
                    "ffmpeg is required for timestamped indexing. Install FFmpeg or pass "
                    "segment_seconds=None to index the entire video as one result."
                )
            spans = [
                (start, min(start + segment_seconds, duration))
                for start in _float_range(0.0, duration, segment_seconds)
            ]
            segments = []
            embeddings = []
            with tempfile.TemporaryDirectory(prefix="vidseek-") as tmpdir:
                for index, (start, end) in enumerate(spans):
                    clip_path = Path(tmpdir) / f"segment-{index:05d}.mp4"
                    _extract_segment(ffmpeg, source, clip_path, start, end - start)
                    embeddings.append(self.encoder.embed_video(clip_path))
                    segments.append((source, start, end))

        metadata_json = self._metadata_json(metadata)
        prepared: list[tuple[str, float, float | None, bytes, int, str]] = []
        for (path, start, end), embedding in zip(segments, embeddings, strict=True):
            blob, dimensions = self._vector_blob(embedding)
            prepared.append((str(path), start, end, blob, dimensions, metadata_json))

        with self._connect() as connection:
            connection.execute("DELETE FROM video_segments WHERE source_path = ?", (str(source),))
            connection.executemany(
                "INSERT INTO video_segments "
                "(source_path, start_seconds, end_seconds, embedding, dimensions, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                prepared,
            )
        return len(prepared)

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed videos by meaning and return cosine-ranked segments.

        ``score`` is cosine similarity, not a calibrated probability.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = np.asarray(self.encoder.embed_text(query.strip()), dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0 or not np.isfinite(query_vector).all():
            raise ValueError("query encoder returned an invalid embedding")
        query_vector = query_vector / query_norm

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_path, start_seconds, end_seconds, embedding, dimensions, metadata_json "
                "FROM video_segments"
            ).fetchall()

        matches: list[dict[str, Any]] = []
        for row in rows:
            if row["dimensions"] != query_vector.size:
                continue
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != query_vector.size:
                continue
            score = float(np.dot(query_vector, vector))
            try:
                metadata = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                metadata = {}
            matches.append(
                {
                    "video_path": row["source_path"],
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                    "score": score,
                    "metadata": metadata,
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]

    def delete_video(self, video_path: str | Path) -> int:
        """Delete all indexed segments for a video and return the number removed."""
        path = str(Path(video_path).expanduser().resolve())
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM video_segments WHERE source_path = ?", (path,)
            )
            return int(cursor.rowcount)

    def count(self) -> int:
        """Return the number of indexed segments."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM video_segments").fetchone()
            return int(row["n"])


def _probe_duration(video_path: Path, *, required: bool) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        if required:
            raise RuntimeError(
                "ffprobe is required for timestamped indexing. Install FFmpeg (which includes "
                "ffprobe) or pass segment_seconds=None."
            )
        return None
    command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        if required:
            raise RuntimeError(f"ffprobe could not read {video_path}: {result.stderr.strip()}")
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        if required:
            raise RuntimeError(f"ffprobe did not return a valid duration for {video_path}")
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def _float_range(start: float, stop: float, step: float) -> Iterable[float]:
    current = start
    while current < stop:
        yield current
        current += step


def _extract_segment(
    ffmpeg: str,
    source: Path,
    destination: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    command = [
        ffmpeg,
        "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_seconds:.3f}",
        "-i", str(source),
        "-t", f"{duration_seconds:.3f}",
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-movflags", "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg failed to extract segment at {start_seconds:.3f}s: {result.stderr.strip()}"
        )
