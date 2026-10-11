"""
Indexer: embeds CodeChunks and stores them for retrieval.

Two modes controlled by EMBEDDING_BACKEND env var:
- "tfidf" (default for deployment): BM25 + TF-IDF, zero RAM, no model download
- "sbert": sentence-transformers, better quality but needs ~400MB RAM

Set EMBEDDING_BACKEND=sbert locally for best results.
Render free tier uses tfidf automatically.
"""

import os
import math
import logging
import hashlib
import re
from pathlib import Path
from typing import Optional
from collections import Counter

from .parser import CodeChunk

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_id(chunk: CodeChunk) -> str:
    content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
    raw = f"{chunk.file_path}::{chunk.name}::{content_hash}"
    return hashlib.md5(raw.encode()).hexdigest()


def _chunk_to_document(chunk: CodeChunk) -> str:
    return f"{chunk.summary}\n\n{chunk.content}"


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return [t for t in tokens if len(t) > 1]


# ── BM25 retriever (zero dependencies, zero RAM) ──────────────────────────────

class BM25Index:
    """Lightweight BM25 implementation — no external dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []
        self.ids: list[str] = []
        self.metadatas: list[dict] = []
        self.tokenized: list[list[str]] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    def add(self, doc_id: str, document: str, metadata: dict):
        tokens = _tokenize(document)
        self.docs.append(document)
        self.ids.append(doc_id)
        self.metadatas.append(metadata)
        self.tokenized.append(tokens)
        for t in set(tokens):
            self.df[t] += 1
        # Recompute avgdl
        total = sum(len(t) for t in self.tokenized)
        self.avgdl = total / len(self.tokenized) if self.tokenized else 1.0

    def query(self, question: str, top_k: int = 6) -> list[dict]:
        if not self.docs:
            return []
        q_tokens = _tokenize(question)
        N = len(self.docs)
        scores = []
        for i, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log((N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
                tf_norm = (tf[t] * (self.k1 + 1)) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                score += idf * tf_norm
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_k]:
            if score > 0:
                results.append({
                    "document": self.docs[i],
                    "metadata": self.metadatas[i],
                    "distance": max(0.0, 1.0 - score / 10.0),  # normalise to ~[0,1]
                })
        return results

    def __len__(self):
        return len(self.docs)


# ── SBERT retriever (better quality, needs ~400MB RAM) ────────────────────────

class SBERTIndex:
    def __init__(self, model_name: str, chroma_path: str, in_memory: bool):
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(model_name, device="cpu")

        if in_memory:
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self.client.get_or_create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[CodeChunk], batch_size: int = 32) -> int:
        ids = [_stable_id(c) for c in chunks]
        existing = set(self.collection.get(ids=ids, include=[])["ids"])
        new_pairs = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
        if not new_pairs:
            return 0

        indexed = 0
        for start in range(0, len(new_pairs), batch_size):
            batch = new_pairs[start: start + batch_size]
            batch_ids = [p[0] for p in batch]
            batch_chunks = [p[1] for p in batch]
            documents = [_chunk_to_document(c) for c in batch_chunks]
            embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()
            metadatas = [{
                "file_path": c.file_path, "language": c.language,
                "chunk_type": c.chunk_type, "name": c.name,
                "start_line": c.start_line, "end_line": c.end_line,
            } for c in batch_chunks]
            self.collection.upsert(
                ids=batch_ids, documents=documents,
                embeddings=embeddings, metadatas=metadatas,
            )
            indexed += len(batch)
        return indexed

    def query(self, question: str, top_k: int = 6) -> list[dict]:
        if self.collection.count() == 0:
            return []
        embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"document": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def __len__(self):
        return self.collection.count()

    def clear(self):
        self.client.delete_collection("codebase")
        self.collection = self.client.get_or_create_collection(
            name="codebase", metadata={"hnsw:space": "cosine"},
        )


# ── Public interface ──────────────────────────────────────────────────────────

class CodebaseIndexer:
    def __init__(
        self,
        chroma_path: str = "./data/chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        in_memory: bool = False,
    ):
        self.embedding_model_name = embedding_model
        force_memory = os.getenv("CHROMA_IN_MEMORY", "").lower() in ("true", "1", "yes")
        self._in_memory = in_memory or force_memory
        self._chroma_path = chroma_path

        # Choose backend based on env var or RAM detection
        backend = os.getenv("EMBEDDING_BACKEND", "tfidf").lower()
        logger.info(f"Embedding backend: {backend}")

        self._backend = backend
        self._bm25: Optional[BM25Index] = None
        self._sbert: Optional[SBERTIndex] = None

        if backend == "sbert":
            self._sbert = SBERTIndex(embedding_model, chroma_path, self._in_memory)
        else:
            self._bm25 = BM25Index()

    def index_chunks(
        self,
        chunks: list[CodeChunk],
        batch_size: int = 32,
        force_reindex: bool = False,
    ) -> int:
        if not chunks:
            return 0

        if self._backend == "sbert" and self._sbert:
            return self._sbert.add_chunks(chunks, batch_size=batch_size)
        else:
            # BM25 — just add all (it's in-memory anyway)
            if self._bm25 is None:
                self._bm25 = BM25Index()
            for chunk in chunks:
                self._bm25.add(
                    doc_id=_stable_id(chunk),
                    document=_chunk_to_document(chunk),
                    metadata={
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type,
                        "name": chunk.name,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                )
            return len(chunks)

    def query(self, question: str, top_k: int = 6) -> list[dict]:
        if self._backend == "sbert" and self._sbert:
            return self._sbert.query(question, top_k)
        elif self._bm25:
            return self._bm25.query(question, top_k)
        return []

    def clear(self):
        if self._sbert:
            self._sbert.clear()
        if self._bm25:
            self._bm25 = BM25Index()

    @property
    def chunk_count(self) -> int:
        if self._backend == "sbert" and self._sbert:
            return len(self._sbert)
        return len(self._bm25) if self._bm25 else 0
