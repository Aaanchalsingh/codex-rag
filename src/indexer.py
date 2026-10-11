"""
Indexer: embeds CodeChunks with sentence-transformers and stores them in ChromaDB.
Supports incremental re-indexing (skips already-indexed chunks by ID).
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .parser import CodeChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "codebase"


def _chunk_to_document(chunk: CodeChunk) -> str:
    """
    Build the text that gets embedded. Including the summary header gives the
    model useful context about what the chunk is.
    """
    return f"{chunk.summary}\n\n{chunk.content}"


def _stable_id(chunk: CodeChunk) -> str:
    """Deterministic ID based on file path + name + content hash."""
    content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
    raw = f"{chunk.file_path}::{chunk.name}::{content_hash}"
    return hashlib.md5(raw.encode()).hexdigest()


class CodebaseIndexer:
    def __init__(
        self,
        chroma_path: str = "./data/chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.chroma_path = str(Path(chroma_path).resolve())
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready at {self.chroma_path} — "
                    f"{self.collection.count()} existing chunks")

    def index_chunks(
        self,
        chunks: list[CodeChunk],
        batch_size: int = 64,
        force_reindex: bool = False,
    ) -> int:
        """
        Embed and store chunks. Skips chunks already in the DB unless force_reindex=True.
        Returns the number of newly indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return 0

        # Build IDs for all chunks
        ids = [_stable_id(c) for c in chunks]

        if not force_reindex:
            # Check which IDs already exist
            existing = self.collection.get(ids=ids, include=[])["ids"]
            existing_set = set(existing)
            new_pairs = [(i, c) for i, c in zip(ids, chunks) if i not in existing_set]
            logger.info(f"{len(existing_set)} chunks already indexed, "
                        f"{len(new_pairs)} new chunks to add.")
        else:
            new_pairs = list(zip(ids, chunks))
            logger.info(f"Force re-indexing {len(new_pairs)} chunks.")

        if not new_pairs:
            return 0

        indexed = 0
        for start in range(0, len(new_pairs), batch_size):
            batch = new_pairs[start: start + batch_size]
            batch_ids = [p[0] for p in batch]
            batch_chunks = [p[1] for p in batch]

            documents = [_chunk_to_document(c) for c in batch_chunks]
            embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()
            metadatas = [
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in batch_chunks
            ]

            self.collection.upsert(
                ids=batch_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            indexed += len(batch)
            logger.info(f"Indexed batch {start // batch_size + 1}: "
                        f"{indexed}/{len(new_pairs)} chunks")

        return indexed

    def query(self, question: str, top_k: int = 6) -> list[dict]:
        """
        Embed the question and retrieve the top_k most relevant chunks.
        Returns list of dicts with 'document', 'metadata', 'distance'.
        """
        embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"document": doc, "metadata": meta, "distance": dist})

        return hits

    def clear(self):
        """Delete all indexed chunks."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Index cleared.")

    @property
    def chunk_count(self) -> int:
        return self.collection.count()
