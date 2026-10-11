"""
Codex RAG — Chat with your codebase.
Tech-savvy UI with light/dark theme, mobile responsive, Groq fallback chain.
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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme init ────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme = st.session_state["theme"]
is_dark = theme == "dark"

# ── CSS ───────────────────────────────────────────────────────────────────────
DARK = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "surface2":   "#21262d",
    "border":     "#30363d",
    "accent":     "#58a6ff",
    "accent2":    "#3fb950",
    "text":       "#e6edf3",
    "text_muted": "#8b949e",
    "user_bubble":"#1f3a5f",
    "bot_bubble": "#1a2332",
    "tag_bg":     "#1c2a1c",
    "tag_color":  "#3fb950",
    "danger":     "#f85149",
    "code_bg":    "#161b22",
}

LIGHT = {
    "bg":         "#ffffff",
    "surface":    "#f6f8fa",
    "surface2":   "#eaeef2",
    "border":     "#d0d7de",
    "accent":     "#0969da",
    "accent2":    "#1a7f37",
    "text":       "#1f2328",
    "text_muted": "#656d76",
    "user_bubble":"#dbeafe",
    "bot_bubble": "#f6f8fa",
    "tag_bg":     "#dafbe1",
    "tag_color":  "#1a7f37",
    "danger":     "#cf222e",
    "code_bg":    "#f6f8fa",
}

C = DARK if is_dark else LIGHT

st.markdown(f"""
<style>
  /* ── Reset & base ─────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
  }}

  /* ── Hide Streamlit chrome ────────────────────────────────── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .stDeployButton {{ display: none; }}
  [data-testid="stToolbar"] {{ display: none; }}

  /* ── App background ───────────────────────────────────────── */
  .stApp {{
    background-color: {C['bg']} !important;
  }}

  /* ── Sidebar ──────────────────────────────────────────────── */
  [data-testid="stSidebar"] {{
    background-color: {C['surface']} !important;
    border-right: 1px solid {C['border']};
  }}
  [data-testid="stSidebar"] * {{
    color: {C['text']} !important;
  }}

  /* ── Inputs ───────────────────────────────────────────────── */
  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stTextArea textarea {{
    background-color: {C['surface2']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
  }}
  .stTextInput > div > div > input:focus {{
    border-color: {C['accent']} !important;
    box-shadow: 0 0 0 3px {C['accent']}22 !important;
  }}

  /* ── Buttons ──────────────────────────────────────────────── */
  .stButton > button {{
    background-color: {C['surface2']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
  }}
  .stButton > button:hover {{
    border-color: {C['accent']} !important;
    color: {C['accent']} !important;
    background-color: {C['accent']}11 !important;
  }}
  [data-testid="baseButton-primary"] > button,
  .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {C['accent']}, {C['accent2']}) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
  }}

  /* ── Slider ───────────────────────────────────────────────── */
  .stSlider [data-baseweb="slider"] {{
    color: {C['accent']} !important;
  }}

  /* ── Metrics ──────────────────────────────────────────────── */
  [data-testid="stMetric"] {{
    background-color: {C['surface2']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
  }}
  [data-testid="stMetricValue"] {{
    color: {C['accent']} !important;
    font-weight: 700 !important;
  }}

  /* ── Chat messages ────────────────────────────────────────── */
  [data-testid="stChatMessage"] {{
    background-color: {C['bot_bubble']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    padding: 16px !important;
  }}
  [data-testid="stChatMessage"][data-testid*="user"] {{
    background-color: {C['user_bubble']} !important;
    border-color: {C['accent']}44 !important;
  }}

  /* ── Chat input ───────────────────────────────────────────── */
  [data-testid="stChatInput"] textarea {{
    background-color: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
  }}
  [data-testid="stChatInput"] textarea:focus {{
    border-color: {C['accent']} !important;
    box-shadow: 0 0 0 3px {C['accent']}22 !important;
  }}

  /* ── Expander ─────────────────────────────────────────────── */
  .streamlit-expanderHeader {{
    background-color: {C['surface2']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text_muted']} !important;
    font-size: 0.85rem !important;
  }}
  .streamlit-expanderContent {{
    background-color: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
  }}

  /* ── Code blocks ──────────────────────────────────────────── */
  code, pre {{
    font-family: 'JetBrains Mono', monospace !important;
    background-color: {C['code_bg']} !important;
    color: {C['accent']} !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
  }}
  pre code {{
    padding: 0 !important;
  }}

  /* ── Divider ──────────────────────────────────────────────── */
  hr {{
    border-color: {C['border']} !important;
    margin: 12px 0 !important;
  }}

  /* ── Custom hero header ───────────────────────────────────── */
  .codex-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 24px 0 8px 0;
  }}
  .codex-logo {{
    font-size: 2.4rem;
    line-height: 1;
  }}
  .codex-title {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, {C['accent']}, {C['accent2']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
  }}
  .codex-subtitle {{
    font-size: 0.9rem;
    color: {C['text_muted']};
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* ── Model badge ──────────────────────────────────────────── */
  .model-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: {C['tag_bg']};
    color: {C['tag_color']};
    border: 1px solid {C['tag_color']}44;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    margin-top: 6px;
  }}

  /* ── Source chip ──────────────────────────────────────────── */
  .source-chip {{
    display: inline-block;
    background-color: {C['surface2']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: {C['accent']};
    margin: 2px 3px 2px 0;
  }}

  /* ── Sidebar brand ────────────────────────────────────────── */
  .sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 12px 0;
  }}
  .sidebar-brand-icon {{
    font-size: 1.6rem;
  }}
  .sidebar-brand-name {{
    font-size: 1.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, {C['accent']}, {C['accent2']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  /* ── Section label ────────────────────────────────────────── */
  .section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {C['text_muted']};
    margin: 16px 0 6px 0;
  }}

  /* ── Fallback chain pills ─────────────────────────────────── */
  .chain-pill {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background-color: {C['surface2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: {C['text_muted']};
  }}
  .chain-pill-num {{
    background: {C['accent']}22;
    color: {C['accent']};
    border-radius: 50%;
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.7rem;
    flex-shrink: 0;
  }}

  /* ── Empty state ──────────────────────────────────────────── */
  .empty-state {{
    text-align: center;
    padding: 48px 24px;
    color: {C['text_muted']};
  }}
  .empty-state-icon {{
    font-size: 3rem;
    margin-bottom: 12px;
  }}
  .empty-state-title {{
    font-size: 1.2rem;
    font-weight: 600;
    color: {C['text']};
    margin-bottom: 6px;
  }}
  .empty-state-sub {{
    font-size: 0.88rem;
    color: {C['text_muted']};
    font-family: 'JetBrains Mono', monospace;
  }}

  /* ── Example prompt cards ─────────────────────────────────── */
  .stButton > button[data-ex="true"] {{
    text-align: left !important;
    background-color: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    color: {C['text_muted']} !important;
    transition: all 0.2s !important;
  }}

  /* ── Footer ───────────────────────────────────────────────── */
  .codex-footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background-color: {C['surface']};
    border-top: 1px solid {C['border']};
    padding: 8px 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: 0.78rem;
    color: {C['text_muted']};
    font-family: 'Inter', sans-serif;
  }}
  .codex-footer a {{
    color: {C['accent']};
    text-decoration: none;
    font-weight: 500;
  }}
  .codex-footer a:hover {{
    text-decoration: underline;
  }}

  /* ── Main area bottom padding (so footer doesn't overlap chat input) ── */
  .main .block-container {{
    padding-bottom: 80px !important;
    max-width: 900px;
  }}

  /* ── Mobile responsive ────────────────────────────────────── */
  @media (max-width: 768px) {{
    .codex-title {{ font-size: 1.4rem; }}
    .codex-logo  {{ font-size: 1.8rem; }}
    .main .block-container {{ padding: 1rem 0.75rem 80px !important; }}
    [data-testid="stChatMessage"] {{ padding: 12px !important; }}
    .codex-footer {{ padding: 8px 12px; font-size: 0.72rem; }}
  }}
</style>
""", unsafe_allow_html=True)

# ── Auto-index on cold start ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_indexer():
    """Load indexer lazily — only called when user visits, not at boot time."""
    try:
        from startup import get_indexer
        return get_indexer()
    except Exception as e:
        print(f"[app] load_indexer failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div class="sidebar-brand">
      <span class="sidebar-brand-icon">⚡</span>
      <span class="sidebar-brand-name">Codex RAG</span>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    col_th1, col_th2 = st.columns([3, 1])
    with col_th1:
        st.markdown(f'<div class="section-label">Appearance</div>', unsafe_allow_html=True)
    with col_th2:
        label = "☀️" if is_dark else "🌙"
        if st.button(label, key="theme_toggle", help="Toggle light/dark theme"):
            st.session_state["theme"] = "light" if is_dark else "dark"
            st.rerun()

    st.divider()

    # API key
    st.markdown('<div class="section-label">Authentication</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Free at https://console.groq.com",
        label_visibility="collapsed",
    )
    if not api_key:
        st.caption("🔑 [Get free API key →](https://console.groq.com)")

    st.divider()

    # Codebase config
    st.markdown('<div class="section-label">Codebase</div>', unsafe_allow_html=True)
    codebase_path = st.text_input(
        "Local Path",
        value=os.getenv("CODEBASE_PATH", ""),
        placeholder="/path/to/your/project",
        label_visibility="visible",
    )
    github_url = st.text_input(
        "GitHub URL",
        value=os.getenv("GITHUB_REPO_URL", ""),
        placeholder="https://github.com/user/repo",
        label_visibility="visible",
    )
    chroma_path = st.text_input(
        "ChromaDB Path",
        value=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        label_visibility="visible",
    )
    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
    )
    top_k = st.slider("Top-k chunks", min_value=3, max_value=12, value=6)

    st.divider()

    # Index controls
    st.markdown('<div class="section-label">Index</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    if col1.button("⚡ Index", use_container_width=True, type="primary"):
        target = github_url.strip() or codebase_path.strip()
        if not target:
            st.error("Enter a path or GitHub URL.")
        else:
            with st.spinner("Parsing & embedding…"):
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
                    st.success(f"✅ {n} new · {indexer.chunk_count} total")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if col2.button("🗑️ Clear", use_container_width=True):
        if "indexer" in st.session_state:
            st.session_state["indexer"].clear()
            st.session_state.pop("qa", None)
            st.success("Cleared.")
        else:
            st.warning("Nothing to clear.")

    # Auto-load index from startup.py (HF Spaces / Render cold start)
    if "indexer" not in st.session_state:
        with st.spinner("⚡ Auto-indexing on startup…"):
            try:
                indexer = load_indexer()
                if indexer and indexer.chunk_count > 0:
                    st.session_state["indexer"] = indexer
                    st.success(f"✅ Auto-indexed {indexer.chunk_count} chunks")
            except Exception as e:
                st.warning(f"Auto-index skipped: {e}")

    if "indexer" in st.session_state:
        st.metric("Chunks indexed", st.session_state["indexer"].chunk_count)

    st.divider()

    # Fallback chain display
    st.markdown('<div class="section-label">LLM Fallback Chain</div>', unsafe_allow_html=True)
    models = [
        ("openai/gpt-oss-120b", "500 t/s"),
        ("openai/gpt-oss-20b",  "1000 t/s"),
        ("qwen/qwen3.8-27b",    "450 t/s"),
    ]
    for i, (m, speed) in enumerate(models, 1):
        st.markdown(f"""
        <div class="chain-pill">
          <span class="chain-pill-num">{i}</span>
          <span style="flex:1">{m}</span>
          <span style="font-size:0.7rem;color:{C['accent2']}">{speed}</span>
        </div>
        """, unsafe_allow_html=True)
    st.caption("Auto-skips decommissioned models")


# ── Main content ──────────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="codex-header">
  <span class="codex-logo">⚡</span>
  <div>
    <div class="codex-title">Codex RAG</div>
    <div class="codex-subtitle">// ask anything about your codebase</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Chat history init
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("model"):
            st.markdown(f'<div class="model-badge">🤖 {msg["model"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📂 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-chip">'
                        f'{src.get("file_path","?")} '
                        f':{src.get("start_line","?")}–{src.get("end_line","?")}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

# Empty state
if not st.session_state["messages"]:
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-state-icon">🔍</div>
      <div class="empty-state-title">Chat with your codebase</div>
      <div class="empty-state-sub">Index a repo, then ask anything below</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Try these")
    examples = [
        "🧩  What are the main components?",
        "🎨  How does the visualization work?",
        "⚙️  What algorithms are implemented?",
        "🔄  How is state managed?",
        "🪝  What does the useGemini hook do?",
        "📐  How is DiagramViewer structured?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_question"] = q.split("  ", 1)[-1]
            st.rerun()

# Handle example click
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
            msg = "⚠️ Add your Groq API key in the sidebar to get started."
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
                    st.markdown(
                        f'<div class="model-badge">🤖 {model_used}</div>',
                        unsafe_allow_html=True,
                    )

                    if sources:
                        with st.expander(f"📂 {len(sources)} source(s) used"):
                            for src in sources:
                                st.markdown(
                                    f'<span class="source-chip">'
                                    f'{src.get("file_path","?")} '
                                    f':{src.get("start_line","?")}–{src.get("end_line","?")}'
                                    f'</span>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "model": model_used,
                    })

                except Exception as e:
                    err = f"❌ {e}"
                    st.error(err)
                    st.session_state["messages"].append({"role": "assistant", "content": err})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="codex-footer">
  <span>⚡ Codex RAG</span>
  <span>·</span>
  <span>© 2026 <strong>Aanchal Singh</strong></span>
  <span>·</span>
  <a href="https://codex-rag.onrender.com" target="_blank">Live Demo</a>
  <span>·</span>
  <a href="https://github.com/Aaanchalsingh/codex-rag" target="_blank">GitHub</a>
  <span>·</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{C['text_muted']}">
    Powered by Groq · ChromaDB · sentence-transformers
  </span>
</div>
""", unsafe_allow_html=True)
