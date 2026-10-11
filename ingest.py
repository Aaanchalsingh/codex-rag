#!/usr/bin/env python3
"""
CLI tool to index a codebase into ChromaDB.

Usage:
    python ingest.py --path ./my-project
    python ingest.py --path ./my-project --extensions .py .ts --force
"""

import sys
import logging
from collections import Counter
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

load_dotenv()

console = Console()


@click.command()
@click.option(
    "--path", "-p",
    required=True,
    help="Path to the codebase directory to index.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--extensions", "-e",
    multiple=True,
    default=None,
    help="File extensions to include, e.g. --extensions .py .ts (defaults to all supported).",
)
@click.option(
    "--chroma-path",
    default="./data/chroma_db",
    show_default=True,
    help="Directory to store the ChromaDB index.",
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="sentence-transformers model name for embeddings.",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Force re-index all chunks even if already in DB.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def main(path, extensions, chroma_path, embedding_model, force, verbose):
    """Index a codebase for the Codex RAG chat tool."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    from src.parser import parse_codebase
    from src.indexer import CodebaseIndexer

    console.rule("[bold blue]Codex RAG — Codebase Indexer")

    repo_path = str(Path(path).resolve())
    exts = list(extensions) if extensions else None

    console.print(f"[cyan]Codebase:[/]        {repo_path}")
    console.print(f"[cyan]Extensions:[/]      {exts or 'all supported'}")
    console.print(f"[cyan]ChromaDB path:[/]   {chroma_path}")
    console.print(f"[cyan]Embedding model:[/] {embedding_model}")
    console.print()

    # ── Parse ──────────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task("Parsing codebase…", total=None)
        chunks = parse_codebase(repo_path, extensions=exts)
        p.update(t, description=f"Parsed [bold green]{len(chunks)}[/] chunks ✓")

    if not chunks:
        console.print("[red]No code chunks found. Check the path and extensions.[/]")
        sys.exit(1)

    # Language breakdown table
    lang_counts = Counter(c.language for c in chunks)
    table = Table(title="Chunks by Language", show_header=True, header_style="bold cyan")
    table.add_column("Language", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for lang, count in lang_counts.most_common():
        table.add_row(lang, str(count))
    console.print(table)
    console.print()

    # ── Index ──────────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as p:
        t1 = p.add_task("Loading embedding model…", total=None)
        indexer = CodebaseIndexer(chroma_path=chroma_path, embedding_model=embedding_model)
        p.update(t1, description="Embedding model loaded ✓", completed=1, total=1)

        t2 = p.add_task("Indexing chunks…", total=len(chunks))
        indexed = indexer.index_chunks(chunks, force_reindex=force)
        p.update(t2, completed=len(chunks))

    console.print()
    console.print(f"[bold green]✓ Done![/] Indexed [bold]{indexed}[/] new chunks.")
    console.print(f"[dim]Total chunks in DB: {indexer.chunk_count}[/]")
    console.print()
    console.print("Now start the chat UI:")
    console.print("[bold]  streamlit run app.py[/]")


if __name__ == "__main__":
    main()
