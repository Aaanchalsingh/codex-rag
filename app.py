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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Codex RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-index on cold start (HF Spaces / first load) ────────────────────────
@st.cache_resource(show_spinner="Loading index…")
def load_indexer():
    from startup import get_indexer
    return get_indexer()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Codex RAG")
    st.caption("Chat with your codebase")
    st.divider()

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Free at https://console.groq.com",
    )

    codebase_path = st.text_input(
        "Codebase Path",
        value=os.getenv("CODEBASE_PATH", ""),
        placeholder="/path/to/your/project",
        help="Local path to index manually.",
    )

    github_url = st.text_input(
        "GitHub Repo URL (optional)",
        value=os.getenv("GITHUB_REPO_URL", ""),
        placeholder="https://github.com/user/repo",
        help="Auto-clone and index this repo.",
    )

    chroma_path = st.text_input(
        "ChromaDB Path",
        value=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
    )

    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
    )

    top_k = st.slider("Retrieved chunks (top-k)", min_value=3, max_value=12, value=6)

    st.divider()
    st.markdown(
        "**Model fallback chain**\n"
        "1. `openai/gpt-oss-120b`\n"
        "2. `openai/gpt-oss-20b`\n"
        "3. `qwen/qwen3.8-27b`\n\n"
        "_Auto-skips decommissioned models._"
    )
    st.divider()

    # ── Index controls ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    if col1.button("⚡ Index", use_container_width=True, type="primary"):
        target = github_url.strip() or codebase_path.strip()
        if not target:
            st.error("Enter a codebase path or GitHub URL.")
        else:
            with st.spinner("Indexing…"):
                try:
                    from src.parser import parse_codebase
                    from src.indexer import CodebaseIndexer
                    import tempfile

                    use_memory = bool(github_url.strip())

                    if github_url.strip():
                        import git
                        tmp = tempfile.mkdtemp(prefix="codex_rag_")
                        git.Repo.clone_from(github_url.strip(), tmp, depth=1)
                        target_path = tmp
                    else:
                        target_path = codebase_path.strip()
                        if not Path(target_path).exists():
                            st.error(f"Path not found: `{target_path}`")
                            st.stop()

                    chunks = parse_codebase(target_path)
                    indexer = CodebaseIndexer(
                        chroma_path=chroma_path,
                        embedding_model=embedding_model,
                        in_memory=use_memory,
                    )
                    n = indexer.index_chunks(chunks)
                    st.session_state["indexer"] = indexer
                    st.success(f"✅ {n} new chunks ({indexer.chunk_count} total)")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if col2.button("🗑️ Clear", use_container_width=True):
        if "indexer" in st.session_state:
            st.session_state["indexer"].clear()
            st.success("Cleared.")
        else:
            st.warning("No index loaded.")

    # Auto-load index from startup.py (HF Spaces cold start)
    if "indexer" not in st.session_state:
        indexer = load_indexer()
        if indexer and indexer.chunk_count > 0:
            st.session_state["indexer"] = indexer

    if "indexer" in st.session_state:
        st.metric("Chunks in index", st.session_state["indexer"].chunk_count)


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Chat with your Codebase")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📂 {len(msg['sources'])} source(s) · `{msg.get('model', '?')}`"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src.get('chunk_type', 'code')}** `{src.get('name', '?')}` "
                        f"— `{src.get('file_path', '?')}` "
                        f"lines {src.get('start_line')}–{src.get('end_line')}"
                    )

# Example prompts
if not st.session_state["messages"]:
    st.markdown("### Try asking:")
    examples = [
        "What are the main components and what do they do?",
        "How does the visualization rendering work?",
        "What algorithms are implemented here?",
        "How is state managed across the app?",
        "What does the useGemini hook do?",
        "How is the DiagramViewer structured?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    st.session_state["messages"].append({"role": "user", "content": q})
    st.rerun()

# Chat input
if question := st.chat_input("Ask anything about your codebase…"):
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            msg = "⚠️ Please enter your Groq API key in the sidebar."
            st.warning(msg)
            st.session_state["messages"].append({"role": "assistant", "content": msg})

        elif "indexer" not in st.session_state:
            msg = "⚠️ No index loaded. Enter a path or GitHub URL and click **⚡ Index**."
            st.warning(msg)
            st.session_state["messages"].append({"role": "assistant", "content": msg})

        else:
            with st.spinner("Searching codebase…"):
                try:
                    from src.qa_chain import CodebaseQA

                    cache_key = f"qa_{api_key[:8]}_{top_k}"
                    if st.session_state.get("qa_cache_key") != cache_key:
                        st.session_state["qa"] = CodebaseQA(
                            indexer=st.session_state["indexer"],
                            groq_api_key=api_key,
                            top_k=top_k,
                        )
                        st.session_state["qa_cache_key"] = cache_key

                    result = st.session_state["qa"].ask(question)
                    answer = result["answer"]
                    sources = result["sources"]
                    model_used = result.get("model", "unknown")

                    st.markdown(answer)
                    st.caption(f"🤖 `{model_used}`")

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
                        "model": model_used,
                    })

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state["messages"].append({"role": "assistant", "content": err})
