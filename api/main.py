"""FastAPI wrapper around answer(), plus the single-page UI.

Phase 4 asks for a rate limit and a question-length cap before this goes
public — both are here, because "add it later" never happens.
"""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import APIStatusError, RateLimitError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ytrag import config
from ytrag.answer import answer as answer_question
from ytrag.answer import search_only
from ytrag.index import stats as index_stats

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model before accepting traffic.

    Without this the model loads lazily on the first question, so the first
    student to use it waits ~14 seconds staring at a spinner while every
    subsequent search takes one. Better to spend that time at startup, in the
    terminal, where a wait is expected and explained.
    """
    from ytrag.embed import get_embedder

    print("Loading embedding model (first run downloads it)...", flush=True)
    embedder = get_embedder()
    print(f"Ready: {embedder.name} ({embedder.dim}-dim)", flush=True)
    yield


app = FastAPI(title="YT Lecture RAG", version="0.1.0", lifespan=lifespan)

# Allow requests from local preview servers, IDE webviews, or external origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    url: str | None = Field(default=None, description="Optional YouTube URL for dynamic on-the-fly ingestion")
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class DynamicQueryRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist link")
    query: str | None = Field(default=None, description="Search query")
    question: str | None = Field(default=None, description="Alternative parameter for query")
    top_k: int = Field(default=5, ge=1, le=20)
    save_index: bool = Field(default=True, description="Whether to update vectors.npz")

    def get_query(self) -> str:
        q = (self.query or self.question or "").strip()
        if not q:
            raise ValueError("Query cannot be empty")
        return q


class IngestOnlyRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist link")
    save_index: bool = Field(default=True, description="Whether to update vectors.npz")


# In-memory cache of dynamic session indices: url -> (vectors, metadata, suggested_questions)
_DYNAMIC_CACHE: dict[str, tuple] = {}


@app.post("/ingest")
def dynamic_ingest_only(payload: IngestOnlyRequest):
    """Explicitly ingest and index a YouTube link (single video or playlist)
    so subsequent queries against it run instantly from memory.
    """
    raw_url = payload.url.strip().strip("'\"")
    if not raw_url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        url = f"https://www.youtube.com/watch?v={raw_url}"
    else:
        url = raw_url

    from ytrag.chunk import ingest_url
    from ytrag.embed import embed_and_save
    from ytrag.answer import generate_suggested_questions

    try:
        if url in _DYNAMIC_CACHE:
            vectors, metadata, questions = _DYNAMIC_CACHE[url]
            return {
                "status": "cached",
                "message": "URL already processed and indexed in cache.",
                "url": url,
                "chunks_count": len(metadata),
                "titles": list({m.get("video_title", "") for m in metadata if m.get("video_title")}),
                "suggested_questions": questions,
            }

        chunks = ingest_url(url)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No audio or speech segments found for the provided YouTube link.",
            )

        save_path = Path("vectors.npz") if payload.save_index else None
        vectors, metadata = embed_and_save(chunks, path=save_path)
        questions = generate_suggested_questions(chunks)
        _DYNAMIC_CACHE[url] = (vectors, metadata, questions)

        titles = list({c.video_title for c in chunks if c.video_title})
        return {
            "status": "success",
            "message": f"Successfully ingested {len(chunks)} chunk(s).",
            "url": url,
            "chunks_count": len(chunks),
            "titles": titles,
            "suggested_questions": questions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")


# A dict of deques is enough for one process on a free tier. Behind more than
# one worker this becomes per-worker, so move it to Redis before it matters.
_HITS: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _HITS[client]

    while window and now - window[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= config.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Slow down — max {config.RATE_LIMIT_REQUESTS} questions per "
            f"{config.RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    window.append(now)


@app.get("/health")
def health():
    return {"status": "ok", "model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL}


@app.get("/stats")
def stats():
    info = index_stats()
    info.pop("videos", None)
    return info


@app.post("/query")
def dynamic_query(payload: DynamicQueryRequest, request: Request):
    """Dynamic RAG endpoint:
    Accepts JSON payload with {"url": "...", "query": "..."}, processes the link
    on-the-fly (video or playlist), chunks it preserving timestamps & URLs, stores
    or updates the dynamic index (vectors.npz), and returns top matching text segments
    along with clickable YouTube links formatted with exact timestamps (&t=120s).
    """
    raw_url = payload.url.strip().strip("'\"")
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        url = f"https://www.youtube.com/watch?v={raw_url}"
    else:
        url = raw_url

    try:
        query = payload.get_query()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    top_k = payload.top_k

    from ytrag.chunk import ingest_url
    from ytrag.embed import embed_and_save, search_embeddings
    from ytrag.answer import generate_suggested_questions

    try:
        # Check in-memory session cache to avoid re-transcribing/re-embedding
        if url in _DYNAMIC_CACHE:
            vectors, metadata, questions = _DYNAMIC_CACHE[url]
        else:
            # 1. Ingest link dynamically (single video or playlist)
            chunks = ingest_url(url)
            if not chunks:
                raise HTTPException(
                    status_code=404,
                    detail="No audio or speech segments found for the provided YouTube link.",
                )

            # 2. Embed and store to index (vectors.npz)
            save_path = Path("vectors.npz") if payload.save_index else None
            vectors, metadata = embed_and_save(chunks, path=save_path)
            questions = generate_suggested_questions(chunks)
            _DYNAMIC_CACHE[url] = (vectors, metadata, questions)

        # 3. Dynamic Vector Search over the index
        results = search_embeddings(query, vectors, metadata, top_k=top_k)

        return {
            "query": query,
            "url": url,
            "count": len(results),
            "results": results,
            "suggested_questions": questions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dynamic ingestion failed: {str(exc)}")


@app.post("/search")
def search(payload: AskRequest, request: Request):
    """Search endpoint. Supports both static index search and on-the-fly dynamic URL search.
    No LLM call, so no quota, no latency, no hallucination.
    """
    if payload.url:
        return dynamic_query(
            DynamicQueryRequest(url=payload.url, query=payload.question, top_k=payload.top_k),
            request,
        )
    return search_only(payload.question, top_k=payload.top_k)


@app.post("/ask")
def ask(payload: AskRequest, request: Request):
    _rate_limit(request)
    try:
        return answer_question(payload.question, top_k=payload.top_k)
    except RateLimitError as exc:
        # The LLM provider's own quota, not ours. Surfacing this as a 500 tells
        # the student nothing; they need to know it is temporary and whose
        # limit it is.
        raise HTTPException(
            status_code=429,
            detail="The language model's usage quota is exhausted. "
            "This is a provider limit, not a problem with your question — try again later.",
        )
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Language model error: {exc.status_code}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/meta")
def meta():
    """Numbers the front page shows. Cheap: no embedding, no LLM."""
    from ytrag.transcribe import cached_video_ids, load_transcript

    ids = cached_video_ids()
    seconds = 0.0
    for vid in ids:
        data = load_transcript(vid)
        if data and data.get("segments"):
            seconds += data["segments"][-1]["end"]
    return {"lectures": len(ids), "hours": round(seconds / 3600, 1)}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
