"""
startup.py — Auto-index a GitHub repo on cold start.

Used by HF Spaces: if GITHUB_REPO_URL is set, this script clones the repo
and indexes it into an in-memory ChromaDB before the Streamlit app starts.

The indexer instance is cached in a module-level variable so app.py can
import it directly without re-indexing on every page refresh.

Usage (called automatically by app.py on first load):
    from startup import get_indexer
    indexer = get_indexer()
"""

import os
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_cached_indexer = None


def get_indexer():
    """
    Returns a ready CodebaseIndexer. Builds it on first call, then caches it.
    
    Priority:
    1. If GITHUB_REPO_URL is set → clone + index into memory
    2. If CHROMA_DB_PATH has existing data → load it  
    3. Otherwise → return empty indexer (user indexes manually via UI)
    """
    global _cached_indexer
    if _cached_indexer is not None:
        return _cached_indexer

    from src.indexer import CodebaseIndexer
    from src.parser import parse_codebase

    github_url = os.getenv("GITHUB_REPO_URL", "").strip()
    chroma_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Option 1: Auto-index from GitHub ─────────────────────────────────────
    if github_url:
        logger.info(f"Auto-indexing from GitHub: {github_url}")
        try:
            import git

            # Clone into a temp directory
            tmp_dir = tempfile.mkdtemp(prefix="codex_rag_")
            logger.info(f"Cloning into {tmp_dir}…")
            git.Repo.clone_from(github_url, tmp_dir, depth=1)

            indexer = CodebaseIndexer(
                embedding_model=embedding_model,
                in_memory=True,  # no disk needed on HF Spaces
            )
            chunks = parse_codebase(tmp_dir)
            n = indexer.index_chunks(chunks)
            logger.info(f"Auto-indexed {n} chunks from {github_url}")
            _cached_indexer = indexer
            return _cached_indexer

        except Exception as e:
            logger.error(f"Auto-index from GitHub failed: {e}")
            # Fall through to local persistent index

    # ── Option 2: Load existing local persistent index ────────────────────────
    local_index = Path(chroma_path)
    if local_index.exists() and any(local_index.iterdir()):
        logger.info(f"Loading existing local index from {chroma_path}")
        indexer = CodebaseIndexer(
            chroma_path=chroma_path,
            embedding_model=embedding_model,
            in_memory=False,
        )
        if indexer.chunk_count > 0:
            _cached_indexer = indexer
            return _cached_indexer

    # ── Option 3: Empty indexer — user indexes manually via UI ───────────────
    logger.info("No pre-built index found. Starting with empty indexer.")
    _cached_indexer = CodebaseIndexer(
        chroma_path=chroma_path,
        embedding_model=embedding_model,
        in_memory=bool(github_url),  # memory mode if we're on HF Spaces
    )
    return _cached_indexer


def reset_indexer():
    """Force a fresh index on next get_indexer() call."""
    global _cached_indexer
    _cached_indexer = None
