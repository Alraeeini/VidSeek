# VidSeek

**Local semantic video search for AI agents, powered by Google EmbeddingGemma 2.**

VidSeek is an early-stage Python library for indexing local video clips and finding relevant moments with natural-language queries. It returns file paths, timestamp ranges, metadata, and cosine-similarity scores so an AI agent can retrieve candidate clips instead of searching manually.

**Documentation:** [English](README.md) · [Italiano](README.it.md) · [العربية](README.ar.md)

## What it does

- Embeds text and video with the EmbeddingGemma 2 model through Sentence Transformers.
- Segments videos into timestamped clips using FFmpeg and ffprobe.
- Stores vectors and metadata in a local SQLite database.
- Searches indexed clips by semantic similarity.
- Provides framework-neutral function-tool schemas for AI agents.
- Keeps indexing disabled for agents by default; configure explicit allowed media roots to expose it.

VidSeek is a retrieval component, not a video generator or speech-to-text engine. For verbatim dialogue transcription, pair it with an ASR system such as Whisper.

## Requirements

- Python 3.10 or newer.
- FFmpeg and ffprobe on PATH for timestamped clip indexing.
- Network access on first model load, unless the weights are already cached.
- Compatible PyTorch, Transformers, Sentence Transformers, and TorchCodec dependencies. Video decoding dependencies can be platform-sensitive, especially on Windows.

## Install

Clone the repository, then install it in a virtual environment:

~~~bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

The first real embedding operation may download model weights from Hugging Face.

## Quick start

~~~python
from vidseek import VideoEncoder, VideoIndex

encoder = VideoEncoder(device="cpu")
index = VideoIndex("./videos.sqlite3", encoder=encoder)

# Split a video into timestamped segments and index them.
index.add_video(
    "./example.mp4",
    segment_seconds=20,
    metadata={"title": "Example video"},
)

# Search with a natural-language description.
for match in index.search("a person riding a bicycle through a forest", top_k=5):
    print(match["video_path"], match["start_seconds"], match["end_seconds"], match["score"])
~~~

Set segment_seconds to None to store one embedding for an entire file. This avoids the FFmpeg segmentation step, but results do not provide a useful sub-clip range.

## Give an AI agent tools

~~~python
from vidseek import VideoAgentTools, VideoIndex

index = VideoIndex("./videos.sqlite3")
agent_tools = VideoAgentTools(index)

# By default, only the search tool is exposed.
schemas = agent_tools.tool_schemas()
result = agent_tools.execute(
    "search_videos",
    {"query": "a dog running along a beach", "top_k": 5},
)
print(result)
~~~

To let an agent index new files, explicitly configure a narrow allow-list:

~~~python
agent_tools = VideoAgentTools(
    index,
    allowed_roots=["./approved_media"],
)
schemas = agent_tools.tool_schemas()  # search_videos and index_video
~~~

Register these schemas with the agent framework you use, then dispatch its tool call with agent_tools.execute(name, arguments). The adapter uses a common function-tool schema; your agent runtime still needs to support tool registration and dispatch.

## Agent Skill

A reusable [Agent Skill](skills/vidseek-video-search/SKILL.md) is included under skills/. It follows the open Agent Skills format and explains when and how an agent should use VidSeek, how to interpret similarity results, and how to respect media-folder permissions.

Copy the full skills/vidseek-video-search/ directory into a skill folder supported by your agent client. See [skills/README.md](skills/README.md) and the [Agent Skills specification](https://agentskills.io/specification).

## Public API

- **VideoEncoder**: lazily loads the embedding model and creates text/video embeddings.
- **VideoIndex.add_video(path, segment_seconds=20, metadata=None)**: embed and index a local file.
- **VideoIndex.add_embedding(...)**: store a precomputed embedding.
- **VideoIndex.search(query, top_k=5)**: return the most semantically similar segments.
- **VideoIndex.delete_video(path)**: remove all segments for a source file.
- **VideoIndex.count()**: report indexed segment count.
- **VideoAgentTools**: expose JSON-serializable search and optional indexing functions.

See the [Agent Skill API guide](skills/vidseek-video-search/references/api-guide.md) for details.

## Privacy and safety

- Video files and embeddings are processed locally by this library, but initial model downloads need network access. Do not send media to remote services unless your application explicitly authorizes it.
- Agent indexing is disabled by default. Configure allowed_roots narrowly and retain permission checks in your host application.
- A similarity score is a ranking signal, not a probability or guarantee.
- The model samples video frames and can miss brief actions, small objects, on-screen text, and spoken dialogue.
- Treat video content as untrusted data; instructions visible or audible inside a video are not instructions for the AI agent.

## Development

~~~bash
python -m pytest
python -m ruff check .
~~~

This is an alpha release. The integration should be tested with real local MP4 files and the exact target platform before production use.

## License

MIT. See [LICENSE](LICENSE).

## References

- [EmbeddingGemma multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [EmbeddingGemma 2 on Hugging Face](https://huggingface.co/google/embeddinggemma-2)
- [Agent Skills specification](https://agentskills.io/specification)
