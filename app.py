"""
Streamlit chat UI for Codex RAG — "chat with your codebase".
Run with: streamlit run app.py
"""

import os
import logging
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Codex RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Codex RAG")
    st.caption("Chat with your codebase")
    st.divider()

    # API key input
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    # Codebase path
    codebase_path = st.text_input(
        "Codebase Path",
        value=os.getenv("CODEBASE_PATH", ""),
        placeholder="/path/to/your/project",
        help="Absolute or relative path to the codebase directory.",
    )

    chroma_path = st.text_input(
        "ChromaDB Path",
        value=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        help="Where the vector index is stored.",
    )

    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        index=0,
        help="Local embedding model — no API needed.",
    )

    top_k = st.slider("Retrieved chunks (top-k)", min_value=3, max_value=12, value=6)

    st.divider()

    # Index button
    if st.button("⚡ Index Codebase", use_container_width=True, type="primary"):
        if not codebase_path:
            st.error("Please enter a codebase path first.")
        elif not Path(codebase_path).exists():
            st.error(f"Path does not exist: `{codebase_path}`")
        else:
            with st.spinner("Parsing & indexing codebase…"):
                try:
                    from src.parser import parse_codebase
                    from src.indexer import CodebaseIndexer

                    chunks = parse_codebase(codebase_path)
                    indexer = CodebaseIndexer(
                        chroma_path=chroma_path,
                        embedding_model=embedding_model,
                    )
                    n = indexer.index_chunks(chunks)
                    st.session_state["indexer"] = indexer
                    st.success(f"✅ Indexed {n} new chunks ({indexer.chunk_count} total)")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    # Show index stats if available
    if "indexer" in st.session_state:
        idx = st.session_state["indexer"]
        st.metric("Chunks in index", idx.chunk_count)

    if st.button("🗑️ Clear Index", use_container_width=True):
        if "indexer" in st.session_state:
            st.session_state["indexer"].clear()
            st.success("Index cleared.")
        else:
            st.warning("No index loaded.")

    st.divider()
    st.markdown(
        "**How to use**\n"
        "1. Paste your Gemini API key\n"
        "2. Enter path to your codebase\n"
        "3. Click **Index Codebase**\n"
        "4. Ask questions below"
    )


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Chat with your Codebase")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📂 {len(msg['sources'])} source(s) used"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src.get('chunk_type', 'code')}** `{src.get('name', '?')}` "
                        f"— `{src.get('file_path', '?')}` "
                        f"lines {src.get('start_line')}–{src.get('end_line')}"
                    )

# Example questions shown when chat is empty
if not st.session_state["messages"]:
    st.markdown("### Try asking:")
    examples = [
        "Where is the authentication logic?",
        "How does the payment flow work?",
        "What does the `process_order` function do?",
        "Which files handle database connections?",
        "How is error handling implemented?",
        "What are the main entry points of this app?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

# Handle example button click
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state["messages"].append({"role": "user", "content": question})
    st.rerun()

# Chat input
if question := st.chat_input("Ask anything about your codebase…"):
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Validate prerequisites
        if not api_key:
            answer = "⚠️ Please enter your Gemini API key in the sidebar."
            st.warning(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        elif "indexer" not in st.session_state:
            if not codebase_path:
                answer = "⚠️ Please enter a codebase path and click **Index Codebase** first."
            else:
                answer = "⚠️ Codebase not indexed yet. Click **Index Codebase** in the sidebar."
            st.warning(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        else:
            with st.spinner("Searching codebase…"):
                try:
                    from src.qa_chain import CodebaseQA

                    # Cache QA chain in session (avoid reloading model each turn)
                    cache_key = f"qa_{api_key[:8]}_{top_k}"
                    if st.session_state.get("qa_cache_key") != cache_key:
                        st.session_state["qa"] = CodebaseQA(
                            indexer=st.session_state["indexer"],
                            gemini_api_key=api_key,
                            top_k=top_k,
                        )
                        st.session_state["qa_cache_key"] = cache_key

                    result = st.session_state["qa"].ask(question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander(f"📂 {len(sources)} source(s) used"):
                            for src in sources:
                                st.markdown(
                                    f"- **{src.get('chunk_type', 'code')}** `{src.get('name', '?')}` "
                                    f"— `{src.get('file_path', '?')}` "
                                    f"lines {src.get('start_line')}–{src.get('end_line')}"
                                )

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state["messages"].append({"role": "assistant", "content": err})
