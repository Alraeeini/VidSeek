"""Framework-neutral tool adapters for calling VidSeek from AI agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .index import VideoIndex


class VideoAgentTools:
    """Expose search/index functions with JSON-serializable responses.

    The schemas use the common OpenAI-style function-tool shape; adapters for
    other agent frameworks can reuse the same ``search_videos`` method.
    """

    def __init__(
        self,
        index: VideoIndex,
        allowed_roots: Iterable[str | Path] | None = None,
    ) -> None:
        self.index = index
        self.allowed_roots = tuple(
            Path(root).expanduser().resolve() for root in (allowed_roots or ())
        )

    def search_videos(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the local video index for clips relevant to a natural-language query."""
        return {"results": self.index.search(query, top_k=top_k)}

    def index_video(
        self,
        video_path: str,
        segment_seconds: float = 20.0,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Index a video under one of the explicitly approved media directories."""
        if not self.allowed_roots:
            raise PermissionError(
                "Agent indexing is disabled until allowed_roots is configured. "
                "This prevents an agent from indexing arbitrary filesystem paths."
            )
        candidate = Path(video_path).expanduser().resolve()
        if not any(_is_within(candidate, root) for root in self.allowed_roots):
            raise PermissionError(f"Video path is outside configured allowed_roots: {candidate}")
        metadata = {"title": title} if title else {}
        count = self.index.add_video(
            str(candidate),
            segment_seconds=segment_seconds,
            metadata=metadata,
        )
        return {"video_path": str(candidate), "indexed_segments": count, "status": "ok"}

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return only tools enabled by this instance's access policy."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_videos",
                    "description": (
                        "Search the local indexed video library by meaning. Returns video paths, "
                        "matching time ranges, metadata, and cosine similarity scores."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural-language description of the desired scene.",
                            },
                            "top_k": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "index_video",
                    "description": (
                        "Index a local video file for later semantic retrieval. Only call for "
                        "paths explicitly authorized by the host application or user."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "video_path": {"type": "string"},
                            "segment_seconds": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "default": 20,
                            },
                            "title": {"type": ["string", "null"]},
                        },
                        "required": ["video_path"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self.allowed_roots:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "index_video",
                        "description": (
                            "Index a local video within an explicitly approved media directory."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "video_path": {"type": "string"},
                                "segment_seconds": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "default": 20,
                                },
                                "title": {"type": ["string", "null"]},
                            },
                            "required": ["video_path"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return tools

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one validated tool call by name."""
        if name == "search_videos":
            return self.search_videos(**arguments)
        if name == "index_video":
            return self.index_video(**arguments)
        raise ValueError(f"Unknown VidSeek tool: {name}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
