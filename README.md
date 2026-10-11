# 🔍 Codex RAG — Chat with Your Codebase

Ask natural language questions about any codebase and get precise, source-cited answers.

```
"Where is the authentication logic?"
"How does the payment flow work?"
"What does process_order do and where is it called?"
```

## How It Works

```
Your codebase
     ↓
Parse files with tree-sitter (extract functions, classes — not arbitrary windows)
     ↓
Embed each chunk locally with sentence-transformers
     ↓
Store in ChromaDB (persisted on disk)
     ↓
User asks question → embed query → find top-k relevant chunks
     ↓
Send chunks + question to Gemini 1.5 Flash → get cited answer
```

## Stack

| Component   | Tool                              | Why                                      |
|-------------|-----------------------------------|------------------------------------------|
| LLM         | Gemini 1.5 Flash (free API)       | 1M token context, great at code          |
| Embeddings  | sentence-transformers (local)     | Free, no API, good code understanding    |
| Vector DB   | ChromaDB (local)                  | Zero setup, persists to disk             |
| Code parser | tree-sitter                       | Splits by functions/classes, not chars   |
| Framework   | LangChain                         | Glues retrieval + LLM together           |
| UI          | Streamlit                         | Fast chat interface                      |

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd codex-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get a **free** Gemini API key at https://aistudio.google.com/app/apikey

### 3. Index your codebase

```bash
python ingest.py --path /path/to/your/project
```

Options:
```
--path, -p       Path to codebase (required)
--extensions     Limit to specific file types, e.g. --extensions .py .ts
--chroma-path    Where to store the index (default: ./data/chroma_db)
--force, -f      Re-index everything even if already indexed
--verbose, -v    Show debug logs
```

### 4. Launch the chat UI

```bash
streamlit run app.py
```

Open http://localhost:8501, paste your API key in the sidebar, and start asking questions.

## Supported Languages

Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, C#, Ruby, PHP

## Project Structure

```
codex-rag/
├── app.py              # Streamlit chat UI
├── ingest.py           # CLI indexer script
├── src/
│   ├── parser.py       # tree-sitter code chunker
│   ├── indexer.py      # ChromaDB embedding store
│   └── qa_chain.py     # LangChain + Gemini QA
├── data/
│   └── chroma_db/      # Persisted vector index (gitignored)
├── requirements.txt
└── .env.example
```

## Tips

- **Re-index after code changes**: run `python ingest.py --path ... --force` to pick up edits.
- **Smaller codebases index faster** — the first run downloads the embedding model (~90MB).
- **Better results**: use `--extensions` to focus on the language you care about.
- The index persists between sessions — you only need to re-index when the code changes.
