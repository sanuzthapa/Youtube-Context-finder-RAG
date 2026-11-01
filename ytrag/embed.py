"""Pluggable embedder.

One backend today (sentence-transformers), but everything downstream talks to
the Protocol, so swapping the model is a config change plus `ytrag reindex`.
"""

import contextlib
import io
from pathlib import Path
import re
import sys
from typing import Protocol

from ytrag.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX

# Noise the model loader prints from a compiled extension, which no env var
# turns off. Filtered rather than suppressed wholesale: anything that is not
# one of these still reaches stderr, so real failures are never hidden.
_BENIGN = re.compile(
    r"unauthenticated requests to the HF Hub|Loading weights:|^\s*$"
)


@contextlib.contextmanager
def _quiet_load():
    """Swallow the known-benign loader chatter, re-emit everything else."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            yield
    finally:
        for line in captured.getvalue().splitlines():
            if not _BENIGN.search(line):
                print(line, file=sys.stderr)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings. Default is bge-m3 (1024-dim, multilingual).

    bge-m3 needs no instruction prefix. Some other models do, and only on the
    query side — bge-*-en-v1.5 wants "Represent this sentence for searching
    relevant passages: ". That is what EMBED_QUERY_PREFIX is for. Getting this
    wrong degrades results silently: no error, just worse answers.
    """

    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        with _quiet_load():
            self.model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 6; keep working on older pins too.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            EMBED_QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Load the embedder once per process — the model is 2.2GB."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER


# ------------------------------------------------------------------
# Index Storage (vectors.npz) & Vector Search
# ------------------------------------------------------------------
def save_index(
    chunks: list,
    vectors: object,
    path: str | Path = "vectors.npz",
) -> Path:
    """Save vector embeddings and associated chunk metadata (including timestamps
    and clickable video URLs) into a compressed .npz archive.
    """
    import json
    from pathlib import Path
    import numpy as np
    from ytrag.models import Chunk

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    vectors_arr = np.array(vectors, dtype=np.float32)

    # Save arrays for direct structured indexing
    video_urls = np.array([getattr(c, "video_url", "") for c in chunks], dtype=object)
    start_times = np.array(
        [int(getattr(c, "start_time", getattr(c, "start_sec", 0))) for c in chunks],
        dtype=np.int32,
    )
    end_times = np.array(
        [int(getattr(c, "end_time", getattr(c, "end_sec", 0))) for c in chunks],
        dtype=np.int32,
    )
    texts = np.array([getattr(c, "text", "") for c in chunks], dtype=object)
    video_titles = np.array([getattr(c, "video_title", "") for c in chunks], dtype=object)
    video_ids = np.array([getattr(c, "video_id", "") for c in chunks], dtype=object)

    payloads = [
        c.to_dict() if hasattr(c, "to_dict") else {
            "chunk_id": getattr(c, "chunk_id", ""),
            "video_id": getattr(c, "video_id", ""),
            "video_title": getattr(c, "video_title", ""),
            "start_time": int(getattr(c, "start_time", getattr(c, "start_sec", 0))),
            "end_time": int(getattr(c, "end_time", getattr(c, "end_sec", 0))),
            "video_url": getattr(c, "video_url", ""),
            "text": getattr(c, "text", ""),
        }
        for c in chunks
    ]

    embedder = get_embedder()
    np.savez_compressed(
        target,
        vectors=vectors_arr,
        video_urls=video_urls,
        start_times=start_times,
        end_times=end_times,
        texts=texts,
        video_titles=video_titles,
        video_ids=video_ids,
        payloads=np.array(json.dumps(payloads)),
        model=np.array(embedder.name),
        dim=np.array(embedder.dim),
    )
    return target


def load_index(path: str | Path = "vectors.npz") -> tuple:
    """Load vector embeddings and metadata from a .npz file."""
    import json
    from pathlib import Path
    import numpy as np

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Index file not found at {target}")

    data = np.load(target, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)

    if "payloads" in data:
        metadata = json.loads(str(data["payloads"]))
    else:
        metadata = []
        n = len(vectors)
        for i in range(n):
            item = {
                "video_url": str(data["video_urls"][i]) if "video_urls" in data else "",
                "start_time": int(data["start_times"][i]) if "start_times" in data else 0,
                "end_time": int(data["end_times"][i]) if "end_times" in data else 0,
                "text": str(data["texts"][i]) if "texts" in data else "",
                "video_title": str(data["video_titles"][i]) if "video_titles" in data else "",
                "video_id": str(data["video_ids"][i]) if "video_ids" in data else "",
            }
            metadata.append(item)

    return vectors, metadata


def embed_chunks(chunks: list, embedder: Embedder | None = None):
    """Generate normalized embeddings for a list of chunks."""
    import numpy as np

    if not chunks:
        embed = embedder or get_embedder()
        return np.empty((0, embed.dim), dtype=np.float32)

    embed = embedder or get_embedder()
    vectors = embed.embed_documents([getattr(c, "text", str(c)) for c in chunks])
    return np.array(vectors, dtype=np.float32)


def search_embeddings(
    query: str,
    vectors: object,
    metadata: list[dict],
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict]:
    """Search vector embeddings via cosine similarity, returning ranked results
    with exact timestamps and clickable links (e.g. &t=120s).
    """
    import numpy as np
    from ytrag.models import format_timestamp

    if len(metadata) == 0 or len(vectors) == 0:
        return []

    embed = embedder or get_embedder()
    q_vec = np.array(embed.embed_query(query), dtype=np.float32)

    # Normalize vectors for cosine similarity
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    v_norms[v_norms == 0] = 1e-9
    normed_vectors = vectors / v_norms

    sims = np.dot(normed_vectors, q_vec)
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = dict(metadata[idx])
        sim = float(sims[idx])
        item["similarity"] = round(sim, 4)
        item["distance"] = round(max(0.0, 1.0 - sim), 4)
        item["score"] = round(sim, 4)

        start_t = int(item.get("start_time", item.get("start_sec", 0)))
        item["start_time"] = start_t
        item["end_time"] = int(item.get("end_time", item.get("end_sec", 0)))
        item["timestamp"] = format_timestamp(start_t)

        vid_id = item.get("video_id", "")
        # Ensure exact timestamp link is populated
        if not item.get("video_url") or "youtu" not in item["video_url"]:
            item["video_url"] = f"https://www.youtube.com/watch?v={vid_id}&t={start_t}s"
        elif "t=" not in item["video_url"]:
            sep = "&" if "?" in item["video_url"] else "?"
            item["video_url"] = f"{item['video_url']}{sep}t={start_t}s"

        results.append(item)

    return results


def embed_and_save(
    chunks: list,
    path: str | Path | None = "vectors.npz",
    embedder: Embedder | None = None,
) -> tuple:
    """Embed chunks, optionally save them to vectors.npz with timestamps & URLs,
    and return (vectors, metadata).
    """
    vectors = embed_chunks(chunks, embedder=embedder)
    metadata = [
        c.to_dict() if hasattr(c, "to_dict") else {
            "chunk_id": getattr(c, "chunk_id", ""),
            "video_id": getattr(c, "video_id", ""),
            "video_title": getattr(c, "video_title", ""),
            "start_time": int(getattr(c, "start_time", getattr(c, "start_sec", 0))),
            "end_time": int(getattr(c, "end_time", getattr(c, "end_sec", 0))),
            "video_url": getattr(c, "video_url", ""),
            "text": getattr(c, "text", ""),
        }
        for c in chunks
    ]
    if path:
        save_index(chunks, vectors, path=path)
    return vectors, metadata


def search_index(
    query: str,
    path: str | Path = "vectors.npz",
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict]:
    """Convenience helper to search an existing vectors.npz directly."""
    vectors, metadata = load_index(path)
    return search_embeddings(query, vectors, metadata, top_k=top_k, embedder=embedder)
