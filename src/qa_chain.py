"""
QA Chain: retrieves relevant code chunks and sends them to Gemini for answering.
Uses LangChain's ChatGoogleGenerativeAI wrapper.
"""

import os
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from .indexer import CodebaseIndexer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert software engineer acting as a codebase assistant.
You answer questions about a codebase using the provided source code context.

Guidelines:
- Be precise and reference specific file paths, function names, and line numbers when available.
- If the answer spans multiple files or functions, explain how they connect.
- If you're not sure, say so — don't hallucinate code that isn't in the context.
- Format code references using backticks.
- Keep answers focused and developer-friendly.
"""

CONTEXT_TEMPLATE = """Relevant code from the codebase:

{context}

---
Question: {question}
"""


def _format_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        score = 1 - hit["distance"]  # cosine similarity (higher = more relevant)
        header = (
            f"### [{i}] {meta.get('chunk_type', 'code').capitalize()} "
            f"`{meta.get('name', '?')}` — {meta.get('file_path', '?')} "
            f"(lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}) "
            f"[relevance: {score:.2f}]"
        )
        parts.append(f"{header}\n```{meta.get('language', '')}\n{hit['document']}\n```")
    return "\n\n".join(parts)


class CodebaseQA:
    def __init__(
        self,
        indexer: CodebaseIndexer,
        gemini_api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        top_k: int = 6,
        temperature: float = 0.2,
    ):
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/app/apikey"
            )

        self.indexer = indexer
        self.top_k = top_k

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
        logger.info(f"QA chain ready with model={model}, top_k={top_k}")

    def ask(self, question: str) -> dict:
        """
        Answer a question about the codebase.

        Returns:
            {
                "answer": str,
                "sources": list of metadata dicts,
                "context": formatted context string,
            }
        """
        if self.indexer.chunk_count == 0:
            return {
                "answer": "The codebase index is empty. Please run the indexer first.",
                "sources": [],
                "context": "",
            }

        # Retrieve relevant chunks
        hits = self.indexer.query(question, top_k=self.top_k)
        if not hits:
            return {
                "answer": "No relevant code found for your question.",
                "sources": [],
                "context": "",
            }

        context = _format_context(hits)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        return {
            "answer": answer,
            "sources": [h["metadata"] for h in hits],
            "context": context,
        }
