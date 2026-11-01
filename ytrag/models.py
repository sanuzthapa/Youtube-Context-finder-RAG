"""The three data structures the whole pipeline moves around."""

import uuid
from dataclasses import dataclass

from ytrag.config import LINK_REWIND_SECONDS

# Fixed namespace so the same chunk_id always produces the same Qdrant point
# ID. Qdrant only accepts UUIDs or unsigned ints as point IDs, so the readable
# "videoid:735" chunk_id gets hashed into a UUID — deterministically, which is
# what makes re-ingesting an upsert instead of a duplicate.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def format_timestamp(seconds: int) -> str:
    """735 -> '12:15', 4335 -> '1:12:15'."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class Video:
    video_id: str
    title: str
    duration: int = 0  # seconds

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class Segment:
    """Raw Whisper output. One utterance, with its own start/end."""

    start: float
    end: float
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Chunk:
    """What actually gets embedded. Several segments merged into a time window."""

    chunk_id: str  # f"{video_id}:{start_sec}" — stable across re-ingest
    video_id: str
    video_title: str
    start_sec: int
    end_sec: int
    text: str
    video_url: str = ""
    start_time: int = 0
    end_time: int = 0

    def __post_init__(self):
        if not self.start_time:
            self.start_time = self.start_sec
        if not self.end_time:
            self.end_time = self.end_sec
        if not self.video_url:
            self.video_url = f"https://www.youtube.com/watch?v={self.video_id}&t={self.start_sec}s"

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, self.chunk_id))

    @property
    def link_sec(self) -> int:
        """Where the citation link should land — slightly before the chunk."""
        return max(0, self.start_sec - LINK_REWIND_SECONDS)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}&t={self.link_sec}s"

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.link_sec)

    def to_payload(self) -> dict:
        """Qdrant / vector store payload. Flat, primitives only."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "video_url": self.video_url,
            "text": self.text,
        }

    def to_dict(self) -> dict:
        """Dictionary representation with all metadata and formatted timestamps."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "video_url": self.video_url,
            "url": self.video_url,
            "timestamp": format_timestamp(self.start_time),
            "text": self.text,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Chunk":
        start_sec = int(payload.get("start_sec", payload.get("start_time", 0)))
        end_sec = int(payload.get("end_sec", payload.get("end_time", 0)))
        video_id = payload["video_id"]
        default_url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"
        return cls(
            chunk_id=payload["chunk_id"],
            video_id=video_id,
            video_title=payload["video_title"],
            start_sec=start_sec,
            end_sec=end_sec,
            text=payload["text"],
            video_url=payload.get("video_url", default_url),
            start_time=int(payload.get("start_time", start_sec)),
            end_time=int(payload.get("end_time", end_sec)),
        )
