"""
QA Chain: retrieves relevant code chunks and sends them to Groq for answering.
Uses a self-healing fallback chain — if a model is decommissioned or overloaded,
it automatically tries the next one.

Fallback order:
  1. openai/gpt-oss-120b  (~500 t/s, best quality)
  2. openai/gpt-oss-20b   (~1000 t/s, fastest)
  3. qwen/qwen3.8-27b     (~450 t/s, good fallback)
"""

import os
import logging
from typing import Optional

from groq import Groq, APIStatusError

from .indexer import CodebaseIndexer

logger = logging.getLogger(__name__)

# Fallback chain — tried in order, skipped on 404 (decommissioned) or 503 (overloaded)
DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

SYSTEM_PROMPT = (
    "You are an expert software engineer acting as a codebase assistant. "
    "Answer questions about the codebase using the provided source code context. "
    "Be precise and reference specific file paths, function names, and line numbers when available. "
    "If the answer spans multiple files, explain how they connect. "
    "If you're unsure, say so. Format code references using backticks."
)

CONTEXT_TEMPLATE = """Relevant code from the codebase:

{context}

---
Question: {question}
"""


def _format_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        score = 1 - hit["distance"]
        header = (
            f"### [{i}] {meta.get('chunk_type', 'code').capitalize()} "
            f"`{meta.get('name', '?')}` — {meta.get('file_path', '?')} "
            f"(lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}) "
            f"[relevance: {score:.2f}]"
        )
        parts.append(f"{header}\n```{meta.get('language', '')}\n{hit['document']}\n```")
    return "\n\n".join(parts)


def _is_retryable(e: APIStatusError) -> bool:
    """Returns True if we should skip to the next model in the chain."""
    return e.status_code in (404, 503, 429, 500)


class CodebaseQA:
    def __init__(
        self,
        indexer: CodebaseIndexer,
        groq_api_key: Optional[str] = None,
        models: Optional[list[str]] = None,
        top_k: int = 6,
        temperature: float = 0.2,
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )

        self.indexer = indexer
        self.top_k = top_k
        self.temperature = temperature
        self.client = Groq(api_key=api_key, http_client=None)

        # Allow env override of model list
        env_models = os.getenv("GROQ_MODELS")
        self.models = models or (
            env_models.split(",") if env_models else DEFAULT_MODELS
        )
        logger.info(f"QA chain ready. Fallback chain: {self.models}")

    def _call_groq(self, prompt: str) -> tuple[str, str]:
        """
        Try each model in the fallback chain.
        Returns (answer, model_used).
        Raises RuntimeError if all models fail.
        """
        last_error = None
        for model in self.models:
            try:
                logger.debug(f"Trying model: {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                return response.choices[0].message.content, model
            except APIStatusError as e:
                if _is_retryable(e):
                    logger.warning(
                        f"Model {model} unavailable (HTTP {e.status_code}), "
                        f"trying next in chain…"
                    )
                    last_error = e
                    continue
                raise  # non-retryable error — bubble up immediately
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}, trying next…")
                last_error = e
                continue

        raise RuntimeError(
            f"All models in fallback chain failed. Last error: {last_error}"
        )

    def ask(self, question: str) -> dict:
        """
        Answer a question about the codebase.
        Returns { "answer": str, "sources": list[dict], "context": str, "model": str }
        """
        if self.indexer.chunk_count == 0:
            return {
                "answer": "The codebase index is empty. Please run the indexer first.",
                "sources": [],
                "context": "",
                "model": None,
            }

        hits = self.indexer.query(question, top_k=self.top_k)
        if not hits:
            return {
                "answer": "No relevant code found for your question.",
                "sources": [],
                "context": "",
                "model": None,
            }

        context = _format_context(hits)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
        answer, model_used = self._call_groq(prompt)

        logger.info(f"Answered via {model_used}")
        return {
            "answer": answer,
            "sources": [h["metadata"] for h in hits],
            "context": context,
            "model": model_used,
        }
