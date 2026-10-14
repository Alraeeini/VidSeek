"""Minimal local demo. The first run downloads model weights from Hugging Face."""

from vidseek import VideoAgentTools, VideoEncoder, VideoIndex

encoder = VideoEncoder(device="cpu", max_frames=32, fps=1.0)
index = VideoIndex("./my_video_library.sqlite3", encoder=encoder)

# Index a video as short clips so search results include timestamp ranges.
index.add_video(
    "./example.mp4",
    segment_seconds=20,
    metadata={"title": "Example video", "source": "local"},
)

# Direct Python API
for result in index.search("a person riding a bicycle", top_k=3):
    print(result)

# Agent adapter: register tools with your chosen AI-agent framework.
tools = VideoAgentTools(index)
print(tools.tool_schemas())
print(tools.search_videos("a person riding a bicycle", top_k=3))
