from __future__ import annotations

import numpy as np
import pytest

from vidseek import VideoIndex


class FakeEncoder:
    def embed_text(self, text: str) -> np.ndarray:
        values = {
            "a cat sleeping": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "a car racing": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        return values[text]

    def embed_video(self, video_path):  # pragma: no cover - add_video needs actual media
        raise NotImplementedError


def test_search_returns_highest_cosine_match_first(tmp_path):
    index = VideoIndex(tmp_path / "test.sqlite3", encoder=FakeEncoder())
    index.add_embedding(
        tmp_path / "cat.mp4",
        [1.0, 0.0, 0.0],
        start_seconds=20,
        end_seconds=40,
        metadata={"title": "Cats"},
    )
    index.add_embedding(
        tmp_path / "car.mp4",
        [0.0, 1.0, 0.0],
        start_seconds=0,
        end_seconds=20,
        metadata={"title": "Cars"},
    )

    results = index.search("a cat sleeping", top_k=2)
    assert len(results) == 2
    assert results[0]["video_path"].endswith("cat.mp4")
    assert results[0]["start_seconds"] == 20
    assert results[0]["metadata"]["title"] == "Cats"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_rejects_empty_query(tmp_path):
    index = VideoIndex(tmp_path / "test.sqlite3", encoder=FakeEncoder())
    with pytest.raises(ValueError, match="non-empty"):
        index.search("   ")


def test_delete_video_removes_segments(tmp_path):
    index = VideoIndex(tmp_path / "test.sqlite3", encoder=FakeEncoder())
    path = tmp_path / "cat.mp4"
    index.add_embedding(path, [1.0, 0.0, 0.0])
    assert index.count() == 1
    assert index.delete_video(path) == 1
    assert index.count() == 0
