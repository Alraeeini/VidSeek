# VidSeek — semantic video search for AI agents

**VidSeek** is an experimental, local-first Python library built on Google's `google/embeddinggemma-2` model. It lets an application index local video files and retrieve relevant clips from natural-language queries. Results include file paths, timestamps, metadata, and a cosine-similarity score, so an AI agent can decide which clips to inspect or open.

> VidSeek is an independent starter project, not an official Google product.

## What it does — and does not do

- Embeds videos and text into a shared semantic vector space using EmbeddingGemma 2.
- Splits videos into timestamped clips before indexing, so results can point to time ranges.
- Stores embeddings and metadata in a local SQLite database.
- Exposes JSON-serializable methods and function-tool schemas for AI agents.
- Works locally after dependencies and model weights are downloaded; it does not call a hosted embedding API.

Embeddings are **not** transcripts, subtitles, detailed captions, or generated summaries. The initial version retrieves relevant clips. Add a speech-to-text model such as Whisper and/or a generative vision-language model if your agent needs exact spoken words, explanations, or full scene descriptions.

## Requirements

- Python 3.10+
- FFmpeg installed and `ffmpeg` plus `ffprobe` available on `PATH` for timestamped segmentation.
- Enough RAM/VRAM for the selected model configuration. First use downloads model weights.
- The model is loaded through Sentence Transformers and Transformers. EmbeddingGemma 2 supports `bfloat16` or `float32`; this library does not force `float16`.

## Install from source

```bash
git clone <your-repository-url>
cd vidseek
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If your platform needs a specific PyTorch build, install the appropriate PyTorch version first using the official PyTorch selector, then install VidSeek.

## Quick start

```python
from vidseek import VideoEncoder, VideoIndex

encoder = VideoEncoder(device="cpu")  # use device="cuda" on a compatible NVIDIA setup
index = VideoIndex("./vidseek.sqlite3", encoder=encoder)

# Creates a separate vector for each 20-second clip.
index.add_video(
    "./my_video.mp4",
    segment_seconds=20,
    metadata={"title": "My video", "project": "demo"},
)

results = index.search("a dog running through a snowy field", top_k=5)
for result in results:
    print(result["video_path"])
    print(result["start_seconds"], result["end_seconds"])
    print(result["score"], result["metadata"])
```

For a single vector per entire video, call `index.add_video("./my_video.mp4", segment_seconds=None)`. Such results will not give useful scene-level timestamps if duration probing is unavailable.

## Use it as AI-agent tools

```python
from vidseek import VideoAgentTools, VideoIndex

index = VideoIndex("./vidseek.sqlite3")
# By default the agent may search, but cannot index arbitrary filesystem paths.
tools = VideoAgentTools(index)

# Use these schemas when registering functions with a compatible agent framework.
schemas = tools.tool_schemas()

# To enable indexing from the agent, explicitly allow a media directory:
# tools = VideoAgentTools(index, allowed_roots=["./approved_media"])
# schemas = tools.tool_schemas()  # now includes index_video

# Dispatch a tool call returned by an agent framework.
result = tools.execute(
    "search_videos",
    {"query": "someone opening a red car door", "top_k": 3},
)
print(result)
```

Available functions:

- `search_videos(query, top_k=5)` — semantic search; returns ranked video segments.
- `index_video(video_path, segment_seconds=20, title=None)` — index a user-authorized local file.

**Security note for agent integrations:** agent-driven indexing is disabled unless `allowed_roots` is explicitly configured. Only files resolved inside those directories can be indexed. A production application should also enforce file size, duration, and codec limits.

## How segmenting works

The default `segment_seconds=20` uses FFmpeg to create temporary video clips, computes one embedding per clip, and records each clip's start/end time. Temporary clips are deleted after indexing. The embeddings and metadata remain in SQLite. Re-indexing a path replaces its old rows after all new embeddings are successfully computed.

EmbeddingGemma 2 samples frames from each clip; it does not inspect every frame at full frame rate. Change `VideoEncoder(fps=..., max_frames=...)` to tune speed versus visual coverage. Smaller segments can improve temporal precision but increase indexing time and database size.

## Tests

The unit tests use a fake encoder and do not download the model:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Current limitations

- Alpha starter implementation; no published PyPI release yet.
- Local video files only; URLs and streaming inputs are not accepted by `VideoIndex.add_video`.
- Search scans the SQLite index linearly. For large collections, add a vector index (FAISS, sqlite-vec, Qdrant, or another suitable backend).
- There is no transcription, diarization, OCR, object tracking, summarization, or generative visual reasoning yet.
- Exact quality and performance depend on the model version, hardware, video content, segment size, and sampling options.

## References

- [EmbeddingGemma 2 multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en)
- [Hugging Face model repository](https://huggingface.co/google/embeddinggemma-2)
