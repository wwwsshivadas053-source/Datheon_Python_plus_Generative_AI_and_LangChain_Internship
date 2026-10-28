import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from config import Config, SAMPLE_DOCS_DIR
from core.document_loader import DocumentLoaderService
from core.text_splitter import TextSplitterService
from core.embedding_manager import EmbeddingManager, LocalSemanticEmbeddings
from core.vector_store import VectorStoreService
from core.rag_pipeline import RAGPipeline


def test_document_loader_pdf():
    pdf_path = SAMPLE_DOCS_DIR / "ai_rag_overview.pdf"
    assert pdf_path.exists(), "Sample PDF must exist."

    docs = DocumentLoaderService.load_pdf(str(pdf_path))
    assert len(docs) >= 3, f"Expected at least 3 pages, found {len(docs)}"
    for idx, doc in enumerate(docs):
        assert doc.metadata["page"] == idx + 1
        assert doc.metadata["file_name"] == "ai_rag_overview.pdf"
        assert len(doc.page_content) > 50


def test_text_splitter():
    pdf_path = SAMPLE_DOCS_DIR / "ai_rag_overview.pdf"
    raw_docs = DocumentLoaderService.load_pdf(str(pdf_path))
    splitter = TextSplitterService(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    assert len(chunks) > len(raw_docs), "Chunking should create multiple chunks per page"
    assert "chunk_id" in chunks[0].metadata
    assert "page" in chunks[0].metadata
    assert chunks[0].metadata["page"] == 1


def test_local_embeddings():
    embeddings = LocalSemanticEmbeddings(n_features=128)
    vecs = embeddings.embed_documents(["Machine learning", "Vector embeddings"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128

    query_vec = embeddings.embed_query("Neural networks")
    assert len(query_vec) == 128


def test_faiss_vector_store_indexing_and_search():
    pdf_path = SAMPLE_DOCS_DIR / "ai_rag_overview.pdf"
    raw_docs = DocumentLoaderService.load_pdf(str(pdf_path))
    splitter = TextSplitterService(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    embeddings = LocalSemanticEmbeddings(n_features=128)
    vector_service = VectorStoreService(store_type="faiss", embeddings=embeddings)
    vector_service.clear()

    added = vector_service.add_documents(chunks)
    assert added == len(chunks)

    results = vector_service.similarity_search_with_score("What is RecursiveCharacterTextSplitter?", k=2)
    assert len(results) == 2
    doc, score = results[0]
    assert 0.0 <= score <= 1.0
    assert "page" in doc.metadata


def test_chroma_vector_store_indexing_and_search():
    pdf_path = SAMPLE_DOCS_DIR / "ai_rag_overview.pdf"
    raw_docs = DocumentLoaderService.load_pdf(str(pdf_path))
    splitter = TextSplitterService(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    embeddings = LocalSemanticEmbeddings(n_features=128)
    vector_service = VectorStoreService(store_type="chroma", embeddings=embeddings)
    vector_service.clear()

    added = vector_service.add_documents(chunks)
    assert added == len(chunks)

    results = vector_service.similarity_search_with_score("What are vector embeddings?", k=2)
    assert len(results) == 2
    doc, score = results[0]
    assert "page" in doc.metadata


def test_rag_pipeline_citations():
    pdf_path = SAMPLE_DOCS_DIR / "ai_rag_overview.pdf"
    raw_docs = DocumentLoaderService.load_pdf(str(pdf_path))
    splitter = TextSplitterService(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    embeddings = LocalSemanticEmbeddings(n_features=128)
    vector_service = VectorStoreService(store_type="faiss", embeddings=embeddings)
    vector_service.clear()
    vector_service.add_documents(chunks)

    rag = RAGPipeline(vector_service=vector_service, llm_provider="local")
    response = rag.query("What is the chunk overlap used in the text splitter?")

    assert "answer" in response
    assert len(response["citations"]) > 0
    assert response["citations"][0]["file_name"] == "ai_rag_overview.pdf"
    assert response["citations"][0]["page"] in [1, 2, 3]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
