"""Run demo queries against the indexed cs-visualizer codebase using Groq."""

from dotenv import load_dotenv
load_dotenv()

from src.indexer import CodebaseIndexer
from src.qa_chain import CodebaseQA

QUERIES = [
    "What are the main components in this project and what do they do?",
    "How does the visualization work? What rendering logic exists?",
    "What algorithms or data structures are implemented here?",
    "How is state managed across the app?",
]

indexer = CodebaseIndexer(chroma_path="./data/chroma_db")
qa = CodebaseQA(indexer=indexer, top_k=5)

print(f"\n{'='*60}")
print(f"Codex RAG — cs-visualizer ({indexer.chunk_count} chunks indexed)")
print(f"Fallback chain: {qa.models}")
print(f"{'='*60}\n")

for i, question in enumerate(QUERIES, 1):
    print(f"\n{'─'*60}")
    print(f"Q{i}: {question}")
    print(f"{'─'*60}")
    result = qa.ask(question)
    print(result["answer"])
    print(f"\n🤖 Model: {result['model']}")
    print(f"📂 Sources: {', '.join(s['file_path'] for s in result['sources'][:3])}")
