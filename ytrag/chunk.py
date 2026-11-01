"""Segment[] -> Chunk[], merged into overlapping time windows.

Deliberately hand-written. A generic text splitter (RecursiveCharacterText-
Splitter and friends) operates on one concatenated string and throws the
timestamps away. The timestamp *is* the product here, so chunking happens in
the time domain, on the segment list, and never splits a segment.
"""

import re
from collections import Counter

from ytrag.config import (
    CHUNK_OVERLAP_SECONDS,
    CHUNK_SECONDS,
    MIN_CHUNK_WORDS,
)
from ytrag.models import Chunk, Segment, Video

_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")


def is_repetitive(text: str, threshold: float = 0.6) -> bool:
    """True if one sentence makes up most of the chunk.

    This is Whisper's loop hallucination surviving into a chunk. These are
    retrieval poison: they match everything and say nothing.
    """
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 3:
        return False
    most_common = Counter(sentences).most_common(1)[0][1]
    return most_common / len(sentences) >= threshold


def chunk_segments(
    video: Video | dict | str,
    segments: list[Segment] | list[dict],
    window_seconds: int = CHUNK_SECONDS,
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """Greedy time-window merge with segment-aligned overlap.
    Preserves exact video_url, start_time, and end_time metadata on every chunk.
    """
    if isinstance(video, str):
        video = Video(video_id=video, title=video)
    elif isinstance(video, dict):
        video = Video(
            video_id=video.get("video_id", video.get("id", "unknown")),
            title=video.get("title", "Untitled Video"),
            duration=int(video.get("duration", 0)),
        )

    normalized_segments: list[Segment] = []
    for s in segments:
        if isinstance(s, Segment):
            normalized_segments.append(s)
        elif isinstance(s, dict):
            normalized_segments.append(
                Segment(
                    start=float(s.get("start", 0)),
                    end=float(s.get("end", 0)),
                    text=str(s.get("text", "")),
                )
            )

    valid_segments = [s for s in normalized_segments if s.text.strip()]
    if not valid_segments:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(valid_segments):
        window_start = valid_segments[i].start
        j = i
        while j < len(valid_segments) and valid_segments[j].end - window_start < window_seconds:
            j += 1
        # j is one past the last segment in the window (or len(valid_segments)).
        last = min(j, len(valid_segments) - 1)
        body_segments = valid_segments[i : last + 1]

        body = " ".join(s.text.strip() for s in body_segments).strip()
        start_sec = int(body_segments[0].start)
        end_sec = int(body_segments[-1].end)
        exact_video_url = f"https://www.youtube.com/watch?v={video.video_id}&t={start_sec}s"

        enough_words = len(body.split()) >= min_words
        if enough_words and not is_repetitive(body):
            chunks.append(
                Chunk(
                    chunk_id=f"{video.video_id}:{start_sec}",
                    video_id=video.video_id,
                    video_title=video.title,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_time=start_sec,
                    end_time=end_sec,
                    video_url=exact_video_url,
                    # Prefixing the title is a cheap trick that meaningfully
                    # improves retrieval: a chunk from minute 34 usually never
                    # restates which topic it belongs to.
                    text=f"{video.title}\n\n{body}",
                )
            )

        if last >= len(valid_segments) - 1:
            break

        # Back up to the nearest segment boundary inside the overlap so the
        # next window re-reads the tail of this one. Never split a segment.
        overlap_start = valid_segments[last].end - overlap_seconds
        next_i = last + 1
        for k in range(i, last + 1):
            if valid_segments[k].start >= overlap_start:
                next_i = k
                break
        # Always make forward progress, even when one segment is longer than
        # the whole window.
        i = max(next_i, i + 1)

    return chunks


def ingest_url(
    url: str,
    window_seconds: int = CHUNK_SECONDS,
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """Dynamic Ingestion Pipeline:
    Accepts a user-submitted YouTube link (either a single video or a playlist),
    resolves its metadata, fetches its transcript (Whisper / cached),
    and chunks it with preserved timestamps and clickable URLs.
    """
    from ytrag.playlist import list_playlist
    from ytrag.transcribe import transcribe

    videos = list_playlist(url)
    if not videos:
        return []

    all_chunks: list[Chunk] = []
    for video in videos:
        segments = transcribe(video)
        chunks = chunk_segments(
            video=video,
            segments=segments,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            min_words=min_words,
        )
        all_chunks.extend(chunks)

    return all_chunks
