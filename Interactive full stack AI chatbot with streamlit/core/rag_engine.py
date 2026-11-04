"""
RAG (Retrieval-Augmented Generation) Engine.
Provides document ingestion (PDF, DOCX, TXT, MD, CSV), semantic chunking,
in-memory vector indexing (TF-IDF Cosine Vectorizer with semantic search),
and source-grounded context augmentation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import io
import re
import csv
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pymupdf  # Modern PyMuPDF import
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

try:
    import docx
except ImportError:
    docx = None


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    source_name: str
    page_number: Optional[int] = None
    character_count: int = 0
    token_estimate: int = 0


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float
    highlight: str


class RAGEngine:
    """
    Self-contained in-memory RAG pipeline.
    Combines resilient multi-format document parsing, sliding-window chunking,
    TF-IDF sublinear-TF vector indexing, and relevancy scoring.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self.documents_metadata: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vector_matrix = None
        self.last_indexed_at: Optional[str] = None

    def clear(self) -> None:
        """Clear all indexed documents and vector index."""
        self.chunks.clear()
        self.documents_metadata.clear()
        self.vectorizer = None
        self.vector_matrix = None
        self.last_indexed_at = None

    def is_indexed(self) -> bool:
        """Return True if documents are loaded and indexed."""
        return len(self.chunks) > 0 and self.vector_matrix is not None

    def get_document_count(self) -> int:
        return len(self.documents_metadata)

    def get_total_chunks(self) -> int:
        return len(self.chunks)

    def extract_text_from_file(self, file_name: str, file_bytes: bytes) -> List[Tuple[str, Optional[int]]]:
        """
        Extract text chunks/pages from uploaded file bytes.
        Returns a list of (text, page_number) tuples.
        """
        lower_name = file_name.lower()
        extracted_pages: List[Tuple[str, Optional[int]]] = []

        if lower_name.endswith(".pdf"):
            if pymupdf is None:
                raise RuntimeError("PyMuPDF (pymupdf) is required to process PDF files.")
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                if page_text:
                    extracted_pages.append((page_text, page_idx + 1))
            doc.close()

        elif lower_name.endswith(".docx"):
            if docx is None:
                raise RuntimeError("python-docx is required to process DOCX files.")
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if full_text:
                extracted_pages.append((full_text, None))

        elif lower_name.endswith(".csv"):
            text_stream = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))
            reader = csv.reader(text_stream)
            rows = []
            header = None
            for row in reader:
                if not header:
                    header = row
                    continue
                row_str = " | ".join(f"{h}: {val}" for h, val in zip(header, row) if val.strip())
                if row_str:
                    rows.append(row_str)
            full_csv_text = "\n".join(rows)
            if full_csv_text:
                extracted_pages.append((full_csv_text, None))

        else:
            # Default to UTF-8 text (txt, md, json, py, html, etc.)
            raw_text = file_bytes.decode("utf-8", errors="ignore").strip()
            if raw_text:
                extracted_pages.append((raw_text, None))

        return extracted_pages

    def _split_text_into_chunks(
        self,
        text: str,
        source_name: str,
        page_num: Optional[int]
    ) -> List[DocumentChunk]:
        """Split raw text into overlapping windowed chunks."""
        chunks: List[DocumentChunk] = []
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return chunks

        start = 0
        text_len = len(cleaned_text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to snap end to a natural boundary (. or newline or space)
            if end < text_len:
                boundary = max(
                    cleaned_text.rfind(". ", start, end),
                    cleaned_text.rfind("! ", start, end),
                    cleaned_text.rfind("? ", start, end),
                    cleaned_text.rfind("\n", start, end)
                )
                if boundary != -1 and boundary > start + (self.chunk_size // 2):
                    end = boundary + 1

            chunk_content = cleaned_text[start:end].strip()
            if len(chunk_content) > 20:  # Skip trivial fragments
                chunk_id = len(self.chunks) + len(chunks) + 1
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    source_name=source_name,
                    page_number=page_num,
                    character_count=len(chunk_content),
                    token_estimate=int(len(chunk_content.split()) * 1.3)
                ))

            if end >= text_len:
                break
            start = end - self.chunk_overlap

        return chunks

    def ingest_file(self, file_name: str, file_bytes: bytes) -> int:
        """
        Ingests a single file, splits it into chunks, and stores metadata.
        Returns the number of new chunks created.
        """
        pages = self.extract_text_from_file(file_name, file_bytes)
        new_chunks: List[DocumentChunk] = []

        for text_content, page_num in pages:
            page_chunks = self._split_text_into_chunks(text_content, file_name, page_num)
            new_chunks.extend(page_chunks)

        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        self.documents_metadata.append({
            "name": file_name,
            "size_bytes": len(file_bytes),
            "chunks_count": len(new_chunks),
            "uploaded_at": datetime.now().isoformat()
        })

        self._build_index()
        return len(new_chunks)

    def _build_index(self) -> None:
        """Build or refresh TF-IDF index across all active chunks."""
        if not self.chunks:
            self.vectorizer = None
            self.vector_matrix = None
            return

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.vector_matrix = self.vectorizer.fit_transform(corpus)
        self.last_indexed_at = datetime.now().isoformat()

    def search(self, query: str, top_k: int = 4, min_score: float = 0.05) -> List[SearchResult]:
        """
        Semantic search over indexed chunks using cosine similarity.
        Returns top-k matching chunks with similarity scores.
        """
        if not self.is_indexed() or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vector_matrix).flatten()

        top_indices = np.argsort(scores)[::-1]
        results: List[SearchResult] = []

        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue

            chunk = self.chunks[idx]
            snippet = chunk.text
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."

            results.append(SearchResult(
                chunk=chunk,
                score=round(score, 4),
                highlight=snippet
            ))

            if len(results) >= top_k:
                break

        return results

    def build_rag_context(self, query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates context prompt injection and source citations list for a query.
        Returns (context_string, sources_metadata_list).
        """
        search_results = self.search(query, top_k=top_k)
        if not search_results:
            return "", []

        context_blocks = []
        sources = []

        for i, res in enumerate(search_results, 1):
            chunk = res.chunk
            page_info = f", Page {chunk.page_number}" if chunk.page_number else ""
            context_blocks.append(
                f"[Source Document {i}: {chunk.source_name}{page_info} | Match Score: {int(res.score * 100)}%]\n"
                f"{chunk.text}"
            )
            sources.append({
                "source": chunk.source_name,
                "page": chunk.page_number,
                "score": res.score,
                "snippet": res.highlight,
                "chunk_id": chunk.chunk_id
            })

        combined_context = (
            "--- RETRIEVED DOCUMENTATION CONTEXT ---\n"
            + "\n\n".join(context_blocks)
            + "\n--- END OF RETRIEVED CONTEXT ---"
        )
        return combined_context, sources

