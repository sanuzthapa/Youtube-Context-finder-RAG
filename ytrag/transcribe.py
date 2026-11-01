"""Audio -> Segment[], cached to JSON.

The cache check at the top of transcribe() is the single most important line
in this codebase. Re-transcribing a 40-video playlist by accident costs hours.
"""

import json
import os
from pathlib import Path

from ytrag.config import (
    TRANSCRIPT_DIR,
    WHISPER_BATCH,
    WHISPER_BEAM,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_LANG,
    WHISPER_MODEL,
)
from ytrag.models import Segment, Video
from ytrag.playlist import delete_audio, download_audio

# Loading large-v3 takes a while, so keep one instance per (model, device).
_MODEL_CACHE: dict[tuple, object] = {}


def _resolve_device() -> tuple[str, str]:
    """Pick device and compute type, preferring CUDA when it actually works."""
    device = WHISPER_DEVICE
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    compute = WHISPER_COMPUTE or ("float16" if device == "cuda" else "int8")
    return device, compute


def get_model(model_name: str = WHISPER_MODEL):
    """Load (and cache) a faster-whisper model, falling back to CPU if CUDA fails.

    A GPU newer than the shipped ctranslate2 build will load and then fail at
    the first inference, so the fallback here is deliberately broad.
    """
    from faster_whisper import WhisperModel

    device, compute = _resolve_device()
    key = (model_name, device, compute)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"[transcribe] CUDA unavailable ({exc}); falling back to CPU int8.")
        device, compute = "cpu", "int8"
        key = (model_name, device, compute)
        model = WhisperModel(model_name, device=device, compute_type=compute)

    _MODEL_CACHE[key] = model
    return model


def get_batched_model(model_name: str = WHISPER_MODEL):
    """Wrap the model in faster-whisper's batched pipeline.

    Same weights, same output quality — it just feeds several VAD-detected
    speech regions through the encoder at once instead of one at a time.
    """
    from faster_whisper import BatchedInferencePipeline

    model = get_model(model_name)
    key = ("batched", id(model))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = BatchedInferencePipeline(model=model)
    return _MODEL_CACHE[key]


def run_whisper(audio_path: str, language: str, model_name: str = WHISPER_MODEL):
    """Transcribe one audio file. Returns (segment_iterator, info).

    vad_filter drops silence outright, and condition_on_previous_text=False
    stops Whisper's loop hallucination — on a long pause or a music sting it
    otherwise repeats the previous phrase over and over. The batched pipeline
    treats each speech region independently, so it never conditions on
    previous text in the first place.
    """
    if WHISPER_BATCH > 0:
        try:
            return get_batched_model(model_name).transcribe(
                audio_path,
                language=language,
                beam_size=WHISPER_BEAM,
                batch_size=WHISPER_BATCH,
                vad_filter=True,
            )
        except (ImportError, AttributeError, TypeError) as exc:
            # Only fall back when this build of faster-whisper genuinely can't
            # batch. Catching more than that would silently drop a long run to
            # the sequential path — four times slower — on a transient error,
            # with nothing in the output to say why it suddenly crawled.
            print(f"[transcribe] batched pipeline unavailable ({exc}); using sequential.")

    return get_model(model_name).transcribe(
        audio_path,
        language=language,
        beam_size=WHISPER_BEAM,
        condition_on_previous_text=False,
        vad_filter=True,
    )


def transcript_path(video_id: str, directory: Path | None = None) -> Path:
    return (directory or TRANSCRIPT_DIR) / f"{video_id}.json"


def load_transcript(video_id: str, directory: Path | None = None) -> dict | None:
    """Return the cached transcript dict, or None if absent/corrupt."""
    path = transcript_path(video_id, directory)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "segments" not in data:
        return None
    return data


def cached_video_ids(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or TRANSCRIPT_DIR).glob("*.json"))


def segments_from_transcript(data: dict) -> list[Segment]:
    return [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for s in data.get("segments", [])
    ]


def _write_transcript(video: Video, language: str, model_name: str, segments: list[Segment]) -> None:
    """Write atomically.

    A Ctrl-C part way through a plain write leaves a truncated JSON file that
    the cache check then trusts forever. Temp file + os.replace makes the
    transcript either fully there or not there at all.
    """
    payload = {
        "video_id": video.video_id,
        "title": video.title,
        "language": language,
        "model": model_name,
        "duration": video.duration,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }
    final = transcript_path(video.video_id)
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, final)


def _parse_vtt(vtt_text: str) -> list[Segment]:
    """Parse WebVTT subtitle text into Segment objects."""
    import re

    time_re = re.compile(
        r"(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})"
    )
    segments = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = time_re.search(line)
        if m:
            def to_sec(h, m_val, s_val, ms_val):
                return (int(h or 0) * 3600) + (int(m_val) * 60) + int(s_val) + (int(ms_val) / 1000.0)

            start = to_sec(
                m.group(1) and m.group(1).rstrip(":"),
                m.group(2),
                m.group(3),
                m.group(4),
            )
            end = to_sec(
                m.group(5) and m.group(5).rstrip(":"),
                m.group(6),
                m.group(7),
                m.group(8),
            )
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not time_re.search(lines[i]):
                clean = re.sub(r"<[^>]+>", "", lines[i].strip())
                if clean and not clean.startswith("NOTE") and not clean.isdigit():
                    text_lines.append(clean)
                i += 1
            if text_lines:
                segments.append(Segment(start=start, end=end, text=" ".join(text_lines)))
            continue
        i += 1
    return segments


def fetch_youtube_captions(video: Video) -> list[Segment] | None:
    """Fetch YouTube's existing manual or auto-generated captions in <1s.
    Avoids multi-minute audio download and Whisper transcription when captions exist.
    """
    import urllib.request
    import yt_dlp
    from ytrag.playlist import _cookie_opts

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    ydl_opts.update(_cookie_opts())

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video.url, download=False)
    except Exception as exc:
        print(f"[transcribe] Fast caption extract note: {exc}; proceeding to next step.")
        return None

    if not info:
        return None

    if not video.duration and info.get("duration"):
        video.duration = int(info["duration"])
    if (not video.title or video.title == video.video_id) and info.get("title"):
        video.title = info["title"]

    subtitles = info.get("subtitles") or {}
    auto_captions = info.get("automatic_captions") or {}

    # Prefer languages: English, Hindi, and common variants
    candidate_langs = ["en", "en-orig", "en-US", "en-GB", "hi", "hi-Latn"]
    all_available = list(subtitles.keys()) + list(auto_captions.keys())
    langs_to_try = [lang for lang in candidate_langs if lang in subtitles or lang in auto_captions]
    if not langs_to_try and all_available:
        langs_to_try = [all_available[0]]

    for lang in langs_to_try:
        tracks = subtitles.get(lang) or auto_captions.get(lang) or []
        selected_track = None
        for fmt in tracks:
            if fmt.get("ext") == "json3":
                selected_track = fmt
                break
        if not selected_track:
            for fmt in tracks:
                if fmt.get("ext") == "vtt":
                    selected_track = fmt
                    break
        if not selected_track and tracks:
            selected_track = tracks[0]

        if not selected_track or not selected_track.get("url"):
            continue

        try:
            req = urllib.request.Request(
                selected_track["url"],
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                content = resp.read().decode("utf-8")

            if selected_track.get("ext") == "json3":
                data = json.loads(content)
                segments = []
                for event in data.get("events", []):
                    segs = event.get("segs", [])
                    text = "".join(s.get("utf8", "") for s in segs).strip()
                    if not text or text == "\n":
                        continue
                    start = float(event.get("tStartMs", 0)) / 1000.0
                    dur = float(event.get("dDurationMs", 0)) / 1000.0
                    end = start + dur
                    segments.append(Segment(start=start, end=end, text=text))
                if segments:
                    return segments
            else:
                segments = _parse_vtt(content)
                if segments:
                    return segments
        except Exception:
            continue

    return None


def transcribe(
    video: Video,
    force: bool = False,
    language: str | None = None,
    model_name: str = WHISPER_MODEL,
    keep_audio: bool = False,
) -> list[Segment]:
    """Transcribe a video, using cached transcript or YouTube captions first."""
    language = language or WHISPER_LANG

    if not force:
        cached = load_transcript(video.video_id)
        if cached is not None:
            return segments_from_transcript(cached)

        # 1. Fast path: Extract YouTube automatic or manual captions (<2s, zero audio download)
        captions = fetch_youtube_captions(video)
        if captions:
            _write_transcript(video, language or "auto", "youtube-captions", captions)
            return captions

    audio_path = download_audio(video, force=force)
    raw_segments, info = run_whisper(str(audio_path), language, model_name)

    segments = [
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in raw_segments
        if s.text and s.text.strip()
    ]

    if not video.duration and getattr(info, "duration", None):
        video.duration = int(info.duration)

    _write_transcript(video, language, model_name, segments)

    if not keep_audio:
        delete_audio(video.video_id)

    return segments
