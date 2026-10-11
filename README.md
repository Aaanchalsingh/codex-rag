---
title: Codex RAG
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

<div align="center">

# ⚡ Codex RAG

### Chat with your codebase using AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-codex--rag.onrender.com-brightgreen?style=for-the-badge&logo=render)](https://codex-rag.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-Aaanchalsingh%2Fcodex--rag-black?style=for-the-badge&logo=github)](https://github.com/Aaanchalsingh/codex-rag)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange?style=for-the-badge)](https://console.groq.com)

> Ask natural language questions about any codebase and get precise, source-cited answers.

```
"Where is the authentication logic?"
"How does the payment flow work?"
"What does process_order do and where is it called?"
```

</div>

---

## 🚀 Live Demo

🔗 **[codex-rag.onrender.com](https://codex-rag.onrender.com)**

---

## ✨ Features

- 🔍 **Semantic code search** — finds relevant code by meaning, not just keywords
- 🌳 **Smart chunking** — splits by functions/classes using tree-sitter, not arbitrary character limits
- ⚡ **Groq-powered** — 500–1000 tokens/second, self-healing fallback chain
- 🎨 **Light & dark theme** — toggle in the sidebar
- 📱 **Mobile responsive** — works on any screen size
- 🔄 **Auto-index from GitHub** — point at any public repo, indexes on startup
- 📂 **Source citations** — every answer shows exact file + line numbers

---

## 🏗️ How It Works

```
Your codebase
     ↓
tree-sitter — parse into functions/classes (not arbitrary windows)
     ↓
sentence-transformers — embed each chunk locally (no API needed)
     ↓
ChromaDB — store vectors in-memory or on disk
     ↓
User asks question → embed → find top-k relevant chunks
     ↓
Groq (gpt-oss-120b → gpt-oss-20b → qwen3.8-27b) → cited answer
```

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| **LLM** | Groq free tier | 500–1000 t/s, self-healing fallback chain |
| **Embeddings** | sentence-transformers (local) | Free, no API, runs on CPU |
| **Vector DB** | ChromaDB | Zero setup, in-memory or persistent |
| **Code parser** | tree-sitter | Structural chunking by syntax, not chars |
| **UI** | Streamlit | Python-native, fast to build |
| **Deployment** | Render (free tier) | No credit card, auto-deploy from GitHub |

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/Aaanchalsingh/codex-rag
cd codex-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add GROQ_API_KEY (free at console.groq.com)
# Optional: GITHUB_REPO_URL to auto-index on startup
```

### 3. Index a codebase

```bash
python ingest.py --path /path/to/your/project
```

### 4. Launch

```bash
streamlit run app.py
```

Open **http://localhost:8501**

---

## 🌐 Deploy on Render (free)

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service → connect your fork
3. Render auto-detects `render.yaml`
4. Add environment variables:
   - `GROQ_API_KEY` — from console.groq.com
   - `GITHUB_REPO_URL` — repo to auto-index on cold start
5. Deploy — get a public URL in ~3 minutes

---

## 📁 Project Structure

```
codex-rag/
├── app.py              # Streamlit UI (light/dark theme, mobile responsive)
├── ingest.py           # CLI indexer
├── startup.py          # Auto-index on cold start (HF Spaces / Render)
├── render.yaml         # Render deployment config
├── src/
│   ├── parser.py       # tree-sitter code chunker
│   ├── indexer.py      # ChromaDB embedding store (local + in-memory)
│   └── qa_chain.py     # Groq QA with self-healing fallback chain
├── requirements.txt
└── .env.example
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Free at console.groq.com |
| `GITHUB_REPO_URL` | Optional | Auto-clone & index on startup |
| `CODEBASE_PATH` | Optional | Local path to index |
| `CHROMA_DB_PATH` | Optional | Where to store index (default: `./data/chroma_db`) |
| `CHROMA_IN_MEMORY` | Optional | `true` for ephemeral (serverless) mode |
| `GROQ_MODELS` | Optional | Comma-separated fallback chain override |

---

<div align="center">

© 2026 **Aanchal Singh** · Built with ⚡ Groq + 🔍 ChromaDB + 🤗 sentence-transformers

</div>
