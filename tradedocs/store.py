"""Doc search. Chroma runs on disk and embeds text with a small model on this
machine, so search needs no API key.

Search is hybrid by default. The embedding search finds chunks with similar
meaning, and a BM25 keyword search finds chunks that share exact words like
"position" or "notional". The two rankings get merged with reciprocal rank
fusion. Set TRADEDOCS_SEARCH=vector to use embeddings alone.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, COLLECTION, TOP_K
from .docs import Chunk

SEARCH_MODE = os.getenv("TRADEDOCS_SEARCH", "hybrid")

_STOPWORDS = set(
    "a an the and or of to in on for is are be can could i my me do does how what which when where why "
    "with it its this that you your from by as at if not so all any get use using want".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOPWORDS]


class BM25:
    """Plain BM25 keyword scoring over a list of documents."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.avg_len = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.freqs = [Counter(t) for t in self.tokens]
        df = Counter(term for toks in self.tokens for term in set(toks))
        n = len(self.tokens)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for toks, freq in zip(self.tokens, self.freqs):
            score = 0.0
            for term in terms:
                if term in freq:
                    f = freq[term]
                    score += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len))
            out.append(score)
        return out


class IndexMissingError(RuntimeError):
    pass


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _embedder():
    return embedding_functions.DefaultEmbeddingFunction()


def build_index(chunks: list[Chunk]) -> int:
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION, embedding_function=_embedder(), metadata={"hnsw:space": "cosine"}
    )
    for start in range(0, len(chunks), 100):
        batch = chunks[start : start + 100]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{"slug": c.slug, "url": c.url, "title": c.title, "section": c.section} for c in batch],
        )
    return collection.count()


def _hit(chunk_id: str, text: str, meta: dict, score: float) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "slug": meta["slug"],
        "url": meta["url"],
        "title": meta["title"],
        "section": meta["section"],
        "score": round(score, 4),
    }


class DocSearch:
    def __init__(self, mode: str = SEARCH_MODE) -> None:
        client = _client()
        if COLLECTION not in [c.name for c in client.list_collections()]:
            raise IndexMissingError("No doc index found. Run `python -m tradedocs.ingest` first.")
        self.collection = client.get_collection(COLLECTION, embedding_function=_embedder())
        self.mode = mode
        everything = self.collection.get(include=["documents", "metadatas"])
        self.ids = everything["ids"]
        self.docs = dict(zip(self.ids, everything["documents"]))
        self.metas = dict(zip(self.ids, everything["metadatas"]))
        self.bm25 = BM25([self.docs[i] for i in self.ids])

    def vector_ranking(self, query: str, n: int) -> list[tuple[str, float]]:
        result = self.collection.query(query_texts=[query], n_results=min(n, len(self.ids)))
        return [(cid, 1 - dist) for cid, dist in zip(result["ids"][0], result["distances"][0])]

    def keyword_ranking(self, query: str, n: int) -> list[tuple[str, float]]:
        scored = sorted(zip(self.ids, self.bm25.scores(query)), key=lambda pair: pair[1], reverse=True)
        return [(cid, s) for cid, s in scored[:n] if s > 0]

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if self.mode == "vector":
            ranked = self.vector_ranking(query, k)
        else:
            # Reciprocal rank fusion: a chunk near the top of either list wins.
            fused: dict[str, float] = {}
            for ranking in (self.vector_ranking(query, 30), self.keyword_ranking(query, 30)):
                for rank, (cid, _) in enumerate(ranking, start=1):
                    fused[cid] = fused.get(cid, 0.0) + 1 / (60 + rank)
            ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:k]
        return [_hit(cid, self.docs[cid], self.metas[cid], score) for cid, score in ranked]
