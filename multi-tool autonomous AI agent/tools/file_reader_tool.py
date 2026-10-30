"""
tools/file_reader_tool.py
=========================
Local file reader tool for the autonomous agent.

Supported file types:
  - Plain text  : .txt, .md, .rst, .log, .csv, .json, .yaml/.yml, .xml,
                  .py, .js, .ts, .html, .css, .sh, .bat, .ini, .cfg, .toml
  - PDF         : .pdf  (via pypdf – optional)
  - Word docs   : .docx (via python-docx – optional)

Safety guardrails:
  - Max file size: 5 MB (configurable via MAX_FILE_SIZE_MB env var)
  - Max preview chars: 8 000 (configurable via MAX_PREVIEW_CHARS)
  - Path traversal detection (blocks "..\" and "../" sequences)
  - Read-only; never writes to disk
"""

import os
import json
import pathlib
from langchain_core.tools import Tool


# --------------------------------------------------------------------------- #
#  Configuration                                                                #
# --------------------------------------------------------------------------- #
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024
MAX_PREVIEW_CHARS: int = int(os.getenv("MAX_PREVIEW_CHARS", "8000"))

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".csv",
    ".json", ".yaml", ".yml", ".xml", ".toml",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".htm", ".css", ".scss",
    ".sh", ".bat", ".ps1",
    ".ini", ".cfg", ".conf", ".env",
    ".c", ".cpp", ".h", ".java", ".go", ".rs",
    ".sql", ".r", ".rb", ".php",
}


# --------------------------------------------------------------------------- #
#  Safety checks                                                                #
# --------------------------------------------------------------------------- #
def _is_safe_path(raw_path: str) -> tuple[bool, str]:
    """Return (is_safe, resolved_absolute_path_or_error_message)."""
    if ".." in raw_path:
        return False, "❌ Security: Path traversal sequences ('..') are not allowed."
    try:
        resolved = pathlib.Path(raw_path).resolve()
    except Exception as exc:
        return False, f"❌ Invalid path: {exc}"
    return True, str(resolved)


# --------------------------------------------------------------------------- #
#  Format helpers                                                               #
# --------------------------------------------------------------------------- #
def _truncate(content: str, max_chars: int, path: str) -> str:
    if len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + f"\n\n⚠️  [Truncated — showing first {max_chars:,} of "
        f"{len(content):,} characters from '{os.path.basename(path)}']"
    )


def _format_output(path: str, content: str, extra_info: str = "") -> str:
    name = os.path.basename(path)
    size_kb = os.path.getsize(path) / 1024
    header = f"File: {name}  |  Size: {size_kb:.1f} KB"
    if extra_info:
        header += f"  |  {extra_info}"
    separator = "-" * 60
    return f"{header}\n{separator}\n{content}"


# --------------------------------------------------------------------------- #
#  Readers per file type                                                        #
# --------------------------------------------------------------------------- #
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return _truncate(content, MAX_PREVIEW_CHARS, path)


def _read_json(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        parsed = json.loads(raw)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pretty = raw   # return as-is if malformed
    return _truncate(pretty, MAX_PREVIEW_CHARS, path)


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "❌ PDF reading requires 'pypdf': run  pip install pypdf"
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"--- Page {i} ---\n{text.strip()}")
    content = "\n\n".join(pages)
    return _truncate(content, MAX_PREVIEW_CHARS, path)


def _read_docx(path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        return "❌ DOCX reading requires 'python-docx': run  pip install python-docx"
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    content = "\n\n".join(paragraphs)
    return _truncate(content, MAX_PREVIEW_CHARS, path)


# --------------------------------------------------------------------------- #
#  Main dispatcher                                                               #
# --------------------------------------------------------------------------- #
def _read_file(raw_path: str) -> str:
    """
    Read a local file and return its content as a string.
    Supports text, JSON, PDF, and DOCX formats.
    """
    raw_path = raw_path.strip().strip('"').strip("'")

    # Safety check
    is_safe, resolved_or_error = _is_safe_path(raw_path)
    if not is_safe:
        return resolved_or_error
    path = resolved_or_error

    # Existence check
    if not os.path.exists(path):
        return (
            f"❌ File not found: '{path}'\n"
            "Tip: Use an absolute path (e.g., C:/Users/you/docs/report.txt) "
            "or a path relative to the project root."
        )

    if not os.path.isfile(path):
        # List directory contents instead
        try:
            entries = os.listdir(path)
            listing = "\n".join(f"  {'[DIR]' if os.path.isdir(os.path.join(path, e)) else '[FILE]'} {e}" for e in entries)
            return f"Directory listing for: {path}\n{listing}"
        except PermissionError:
            return f"❌ Permission denied: cannot list directory '{path}'"

    # Size check
    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE_BYTES:
        return (
            f"❌ File too large: {size / 1024 / 1024:.1f} MB "
            f"(limit is {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB). "
            "Set MAX_FILE_SIZE_MB in .env to increase the limit."
        )

    ext = pathlib.Path(path).suffix.lower()

    try:
        if ext == ".pdf":
            content = _read_pdf(path)
        elif ext == ".docx":
            content = _read_docx(path)
        elif ext == ".json":
            content = _read_json(path)
        elif ext in TEXT_EXTENSIONS or size < 512_000:
            content = _read_text(path)
        else:
            return (
                f"⚠️  Unsupported file type: '{ext}'. "
                "Supported: text files, .json, .pdf, .docx\n"
                f"Full path: {path}"
            )
    except PermissionError:
        return f"❌ Permission denied: cannot read '{path}'"
    except Exception as exc:
        return f"❌ Error reading file: {exc}"

    return _format_output(path, content)


# --------------------------------------------------------------------------- #
#  LangChain Tool factory                                                       #
# --------------------------------------------------------------------------- #
def get_file_reader_tool() -> Tool:
    """Returns a LangChain Tool that reads local files."""
    return Tool(
        name="file_reader",
        func=_read_file,
        description=(
            "Reads the content of a local file on disk. "
            "Input: an absolute or relative file path string "
            "(e.g., 'C:/Users/you/report.txt' or 'data/notes.md'). "
            "Supported formats: plain text, Markdown, CSV, JSON, YAML, Python, "
            "HTML, PDF (.pdf), and Word documents (.docx). "
            "If a directory path is given, lists its contents. "
            "Max file size: 5 MB. "
            "Output: the file's text content (truncated if very large). "
            "Use when the user asks to read, summarize, or analyze a local file."
        ),
    )
