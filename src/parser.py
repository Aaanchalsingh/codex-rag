"""
Code parser using tree-sitter for intelligent chunking.
Splits code into meaningful units: functions, classes, methods — not arbitrary character windows.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Supported languages and their file extensions
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
}

# Node types that represent meaningful code units per language
CHUNK_NODE_TYPES = {
    "python": [
        "function_definition",
        "async_function_definition",
        "class_definition",
    ],
    "javascript": [
        "function_declaration",
        "arrow_function",
        "method_definition",
        "class_declaration",
        "function_expression",
    ],
    "typescript": [
        "function_declaration",
        "arrow_function",
        "method_definition",
        "class_declaration",
        "function_expression",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "java": [
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
    "rust": [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
    ],
}


@dataclass
class CodeChunk:
    """A meaningful chunk of code extracted from a source file."""
    content: str           # The actual code text
    file_path: str         # Relative path to the source file
    language: str          # Programming language
    chunk_type: str        # e.g. "function", "class", "module"
    name: str              # Name of the function/class if extractable
    start_line: int
    end_line: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}::{self.name}::{self.start_line}"

    @property
    def summary(self) -> str:
        return (
            f"[{self.language}] {self.chunk_type} `{self.name}` "
            f"in {self.file_path} (lines {self.start_line}-{self.end_line})"
        )


def _try_treesitter_parse(source_bytes: bytes, language: str) -> Optional[list]:
    """
    Attempt to parse with tree-sitter. Handles both 0.21.x and 0.23.x APIs.
    Returns list of matching nodes, or None if tree-sitter unavailable.
    """
    try:
        from tree_sitter import Language, Parser
        import tree_sitter

        # Build language object — API differs between 0.21 and 0.23
        ts_version = tuple(int(x) for x in tree_sitter.__version__.split(".")[:2])

        def _get_lang(module_name: str, attr: Optional[str] = None):
            import importlib
            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr or "language")
            raw = fn()
            if ts_version >= (0, 22):
                return Language(raw)
            else:
                # 0.21 requires Language(capsule, name) form
                return Language(raw, module_name.replace("tree_sitter_", ""))

        lang_builders = {
            "python":     lambda: _get_lang("tree_sitter_python"),
            "javascript": lambda: _get_lang("tree_sitter_javascript"),
            "typescript": lambda: _get_lang("tree_sitter_typescript", "language_typescript"),
            "java":       lambda: _get_lang("tree_sitter_java"),
            "go":         lambda: _get_lang("tree_sitter_go"),
            "rust":       lambda: _get_lang("tree_sitter_rust"),
            "cpp":        lambda: _get_lang("tree_sitter_cpp"),
        }

        if language not in lang_builders:
            return None

        lang = lang_builders[language]()

        if ts_version >= (0, 22):
            parser = Parser(lang)
        else:
            parser = Parser()
            parser.set_language(lang)

        tree = parser.parse(source_bytes)

        target_types = set(CHUNK_NODE_TYPES.get(language, []))
        results = []

        def walk(node):
            if node.type in target_types:
                results.append(node)
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return results

    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"tree-sitter parse failed for {language}: {e}")
        return None


def _extract_name(node, source_bytes: bytes, language: str) -> str:
    """Try to extract the name identifier from a tree-sitter node."""
    # Look for an 'identifier' or 'name' child node
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return "<anonymous>"


def parse_file(file_path: str, repo_root: str) -> list[CodeChunk]:
    """
    Parse a single source file into CodeChunks.
    Uses tree-sitter when available; falls back to whole-file chunking.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    language = LANGUAGE_MAP.get(ext)

    if not language:
        return []

    try:
        source = path.read_bytes()
        source_text = source.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return []

    relative_path = os.path.relpath(file_path, repo_root)
    chunks = []

    # Try tree-sitter structural parsing
    nodes = _try_treesitter_parse(source, language)

    if nodes:
        for node in nodes:
            code_text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            name = _extract_name(node, source, language)
            chunk_type = _normalize_type(node.type)

            chunks.append(CodeChunk(
                content=code_text,
                file_path=relative_path,
                language=language,
                chunk_type=chunk_type,
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                metadata={"node_type": node.type},
            ))

        # If tree-sitter found nothing meaningful, fall through to fallback
        if chunks:
            return chunks

    # Fallback: split file into ~60-line overlapping windows
    lines = source_text.splitlines()
    window, overlap = 60, 10
    for i in range(0, max(1, len(lines)), window - overlap):
        chunk_lines = lines[i: i + window]
        if not chunk_lines:
            break
        chunks.append(CodeChunk(
            content="\n".join(chunk_lines),
            file_path=relative_path,
            language=language,
            chunk_type="module",
            name=f"lines_{i+1}_{i+len(chunk_lines)}",
            start_line=i + 1,
            end_line=i + len(chunk_lines),
            metadata={"fallback": True},
        ))

    return chunks


def _normalize_type(node_type: str) -> str:
    if "class" in node_type or "struct" in node_type or "interface" in node_type:
        return "class"
    if "function" in node_type or "method" in node_type or "arrow" in node_type:
        return "function"
    return "declaration"


def parse_codebase(repo_path: str, extensions: Optional[list[str]] = None) -> list[CodeChunk]:
    """
    Walk an entire codebase directory and return all CodeChunks.

    Args:
        repo_path: Root directory of the codebase to index.
        extensions: Optional list of extensions to include (e.g. ['.py', '.ts']).
                    Defaults to all supported languages.
    """
    repo_path = str(Path(repo_path).resolve())
    allowed_exts = set(extensions) if extensions else set(LANGUAGE_MAP.keys())

    all_chunks: list[CodeChunk] = []
    skipped_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                    "dist", "build", ".next", "target", "vendor"}

    for dirpath, dirnames, filenames in os.walk(repo_path):
        # Prune directories we never want to descend into
        dirnames[:] = [d for d in dirnames if d not in skipped_dirs and not d.startswith(".")]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in allowed_exts:
                continue
            full_path = os.path.join(dirpath, fname)
            try:
                file_chunks = parse_file(full_path, repo_path)
                all_chunks.extend(file_chunks)
                logger.debug(f"Parsed {full_path}: {len(file_chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to parse {full_path}: {e}")

    logger.info(f"Total chunks extracted: {len(all_chunks)} from {repo_path}")
    return all_chunks
