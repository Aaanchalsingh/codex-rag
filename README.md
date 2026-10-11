---
title: Codex RAG
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# 🔍 Codex RAG — Chat with Your Codebase

Ask natural language questions about any codebase and get precise, source-cited answers.

```
"Where is the authentication logic?"
"How does the payment flow work?"
"What does process_order do and where is it called?"
```

## How It Works

```
Your codebase (cloned from GitHub)
     ↓
Parse files with tree-sitter (extract functions, classes)
     ↓
Embed each chunk locally with sentence-transformers
     ↓
Store in ChromaDB (in-memory)
     ↓
User asks question → embed query → find top-k relevant chunks
     ↓
Send chunks + question to Groq → get cited answer
```

## Stack

| Component   | Tool                              | Why                                      |
|-------------|-----------------------------------|------------------------------------------|
| LLM         | Groq (free API)                   | 500-1000 t/s, self-healing fallback chain|
| Embeddings  | sentence-transformers (local)     | Free, no API, good code understanding    |
| Vector DB   | ChromaDB (in-memory)              | Zero setup, fast                         |
| Code parser | tree-sitter                       | Splits by functions/classes, not chars   |
| UI          | Streamlit                         | Fast chat interface                      |

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Aaanchalsingh/codex-rag
cd codex-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
# Add your GITHUB_REPO_URL to auto-index on startup
```

### 3. Run

```bash
streamlit run app.py
```

## Environment Variables (for HF Spaces)

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Free at console.groq.com |
| `GITHUB_REPO_URL` | Optional | Auto-index this repo on startup |

## Project Structure

```
codex-rag/
├── app.py              # Streamlit chat UI
├── ingest.py           # CLI indexer script
├── startup.py          # Auto-index on cold start
├── src/
│   ├── parser.py       # tree-sitter code chunker
│   ├── indexer.py      # ChromaDB embedding store
│   └── qa_chain.py     # Groq QA with fallback chain
├── requirements.txt
└── .env.example
```
