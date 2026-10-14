"""EmbeddingGemma 2 wrapper for text-to-video semantic search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class VideoEncoder:
    """Create text and video embeddings with Google's EmbeddingGemma 2.

    The model is loaded lazily, so importing :mod:`vidseek` does not immediately
    download model weights. Video encoding requires the multimedia dependencies
    listed in the project README.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        device: str | None = None,
        load_audio_encoder: bool = False,
        max_frames: int = 32,
        fps: float | None = 1.0,
        add_timestamps: bool = True,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be greater than 0, or None")

        self.model_name = model_name
        self.device = device
        self.load_audio_encoder = load_audio_encoder
        self.max_frames = max_frames
        self.fps = fps
        self.add_timestamps = add_timestamps
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on installation
                raise RuntimeError(
                    "Missing model dependencies. Install with: "
                    "pip install -e '.[dev]'"
                ) from exc

            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            # Keep text + vision/video encoders, and omit the separate audio
            # encoder by default to reduce memory use for a video-first library.
            if not self.load_audio_encoder:
                kwargs["config_kwargs"] = {"audio_config": None}

            try:
                self._model = SentenceTransformer(self.model_name, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load {self.model_name!r}. Check your internet connection, "
                    "the installed Transformers/Sentence-Transformers versions, available RAM, "
                    "and access to the model repository."
                ) from exc
        return self._model

    @staticmethod
    def _normalize(vector: Any) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if not np.isfinite(array).all() or norm == 0.0:
            raise ValueError("The model returned an invalid or zero-length embedding")
        return array / norm

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a natural-language search query into the shared vector space."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        model = self._get_model()
        vector = model.encode(text.strip(), normalize_embeddings=True)
        return self._normalize(vector)

    def embed_video(self, video_path: str | Path) -> np.ndarray:
        """Embed a local video file into the shared vector space.

        The encoder samples frames according to the configured FPS and frame cap.
        For precise timestamp retrieval in longer videos, index short segments
        with :class:`vidseek.VideoIndex` instead of embedding the entire file once.
        """
        path = Path(video_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        model = self._get_model()
        processing_kwargs = {
            "video": {
                "max_frames": self.max_frames,
                "fps": self.fps,
                "overflow_strategy": "uniform",
                "add_timestamps": self.add_timestamps,
            }
        }
        try:
            vector = model.encode(
                {"video": str(path.resolve())},
                normalize_embeddings=True,
                processing_kwargs=processing_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not embed video {str(path)!r}. Ensure the file is readable and "
                "the video dependencies (including TorchCodec/FFmpeg support) are installed."
            ) from exc
        return self._normalize(vector)
