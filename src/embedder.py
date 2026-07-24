"""Embedder abstraction (DECISIONS D-015) over the resolved embedder (D-008).

Layers, kept distinct on purpose:
  - Embedder interface: embed_documents / embed_query, exposing model_id + dim. Hides the
    doc/query ASYMMETRY each backend needs (nomic prefixes vs Voyage input_type).
  - EmbeddingCache: keyed by (model_id, role, sha256(text)). This is what actually makes
    an embedder swap cheap -- a new model_id => cache miss => auto re-embed. The interface
    makes the CODE swap cheap; it does NOT make old vectors survive a model change (different
    vector space => full re-embed, per the D-008 guardrail). Do not conflate the two.

Resolved backend: nomic-embed-text-v1.5, dim=768, window=8192, cosine over L2-normalized
vectors (FORKS.md Default d). Voyage-3 remains a real, unused fallback (D-008) -- stubbed
so the interface is honest without building the path we didn't take.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_ROOT = Path("data/cache/emb")

# CPU-only, 268 short docs. Larger batches give little speedup and risk memory pressure.
# TUNABLE(CPU batch; revisit if corpus-embed wall-time is painful or memory spikes)
DEFAULT_BATCH_SIZE = 32


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """On-disk vector cache: data/cache/emb/<model_id>/<role>/<sha256>.npy."""

    def __init__(self, model_id: str, root: Path = CACHE_ROOT):
        self.base = root / model_id.replace("/", "_")

    def _path(self, role: str, text: str) -> Path:
        return self.base / role / f"{_key(text)}.npy"

    def get(self, role: str, text: str) -> np.ndarray | None:
        p = self._path(role, text)
        return np.load(p) if p.exists() else None

    def put(self, role: str, text: str, vec: np.ndarray) -> None:
        p = self._path(role, text)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, vec.astype(np.float32))


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class NomicLocal:
    """nomic-embed-text-v1.5 via sentence-transformers (local, offline after first load).

    nomic REQUIRES task prefixes; getting them wrong silently craters retrieval, so they
    live here and nowhere else. Returns L2-normalized float32 (cosine == dot product).
    """

    model_id = "nomic-embed-text-v1.5"
    dim = 768
    _DOC_PREFIX = "search_document: "
    _QUERY_PREFIX = "search_query: "

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, use_cache: bool = True):
        # Lazy model load: the sentence-transformers/torch import is deferred to the first cache
        # MISS (_ensure_model), so a fully-cached run -- e.g. re-scoring the eval queries, whose
        # vectors are already on disk -- never loads torch. Concrete on this machine, where
        # Windows Application Control blocks torch's DLLs (the same wall as spaCy, D-005): cached
        # diagnostics still run. This is the D-015 cache paying off -- a full hit needs no model.
        self.batch_size = batch_size
        self._cache = EmbeddingCache(self.model_id) if use_cache else None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(f"nomic-ai/{self.model_id}", trust_remote_code=True)
        return self._model

    def _encode(self, prefixed: list[str]) -> np.ndarray:
        return self._ensure_model().encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _embed(self, texts: list[str], role: str, prefix: str) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        misses: list[int] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(role, t) if self._cache else None
            if cached is None:
                misses.append(i)
            else:
                out[i] = cached
        if misses:
            fresh = self._encode([prefix + texts[i] for i in misses])
            for j, i in enumerate(misses):
                out[i] = fresh[j]
                if self._cache:
                    self._cache.put(role, texts[i], fresh[j])
        return np.vstack(out)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "document", self._DOC_PREFIX)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "query", self._QUERY_PREFIX)[0]


class VoyageAPI:
    """Fallback embedder (D-008), unused because nomic loaded. Stub kept so the interface
    is real; build out only if we ever switch (needs `voyageai` + VOYAGE_API_KEY, dim=1024,
    input_type='document'/'query')."""

    model_id = "voyage-3"
    dim = 1024

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Voyage fallback not built: nomic-embed-text-v1.5 loaded locally (D-008). "
            "Build this only on an embedder switch; needs voyageai + VOYAGE_API_KEY."
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def load_embedder(name: str = "nomic", **kw) -> Embedder:
    if name == "nomic":
        return NomicLocal(**kw)
    if name == "voyage":
        return VoyageAPI(**kw)
    raise ValueError(f"unknown embedder: {name!r}")
